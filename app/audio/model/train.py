from pathlib import Path
import sys

# Ensure 'app' directory is in sys.path
_current = Path(__file__).resolve()
for _p in [_current.parent, _current.parent.parent, _current.parent.parent.parent]:
    if _p.name == "app" and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
    elif (_p / "app").is_dir() and str(_p / "app") not in sys.path:
        sys.path.insert(0, str(_p / "app"))

import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR



try:
    from audio.model.dataset import make_dataloaders
    from audio.model.backbone import build_model
    from audio.model.loss import build_loss
    from audio import CLIPS_METADATA_PATH, CHECKPOINTS_DIR
except ImportError:
    from model.dataset import make_dataloaders
    from model.backbone import build_model
    from model.loss import build_loss
    from audio import CLIPS_METADATA_PATH, CHECKPOINTS_DIR



def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        if targets.ndim > 1:
            target_labels = targets.argmax(dim=1)
        else:
            target_labels = targets

        _, pred_top5 = logits.topk(5, dim=1, largest=True, sorted=True)
        top1_correct = (pred_top5[:, 0] == target_labels).sum().item()
        top5_correct = (pred_top5 == target_labels.unsqueeze(1)).any(dim=1).sum().item()
        n = len(targets)
        return {
            "top1_acc": top1_correct / max(n, 1),
            "top5_acc": top5_correct / max(n, 1),
        }


def train_epoch(
    model, loader, criterion, optimizer, scaler, device
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    num_batches = len(loader)

    for i, batch in enumerate(loader):
        specs = batch["spectrogram"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(specs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metrics = compute_metrics(logits, labels)
        total_loss += loss.item() * len(specs)
        total_top1 += metrics["top1_acc"] * len(specs)
        total_top5 += metrics["top5_acc"] * len(specs)

        # ── Print live progress every 50 batches ──
        if (i + 1) % 50 == 0 or (i + 1) == num_batches:
            print(
                f"  Batch [{i + 1:04d}/{num_batches}] | Batch Loss: {loss.item():.4f}"
            )

    n = len(loader.dataset)
    return total_loss / n, total_top1 / n, total_top5 / n


def validate(model, loader, criterion, device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    with torch.no_grad():
        for batch in loader:
            specs = batch["spectrogram"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(specs)
                loss = criterion(logits, labels)
            metrics = compute_metrics(logits, labels)
            total_loss += loss.item() * len(specs)
            total_top1 += metrics["top1_acc"] * len(specs)
            total_top5 += metrics["top5_acc"] * len(specs)
    n = len(loader.dataset)
    return total_loss / n, total_top1 / n, total_top5 / n


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train Bird Sound Classifier")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--backbone", type=str, default="efficientnet_v2_s", help="Backbone model")
    parser.add_argument("--clips-csv", type=str, default=None, help="Path to clips_metadata.csv")
    parser.add_argument("--save-dir", type=str, default=None, help="Checkpoint save directory")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--no-mixup", action="store_true", help="Disable acoustic mixup")
    return parser


def main(args=None):
    if args is None:
        parser = parse_args()
        # Parse known args so calling in environments like notebooks won't fail
        args, _ = parser.parse_known_args()

    # ── Config ─────────────────────────────────────────────────────────────
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers if args.num_workers is not None else min(4, (import_os := __import__('os')).cpu_count() or 2)
    EPOCHS = args.epochs
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    BACKBONE = args.backbone
    USE_MIXUP = not args.no_mixup
    SAVE_DIR = Path(args.save_dir) if args.save_dir else CHECKPOINTS_DIR
    CLIPS_CSV = Path(args.clips_csv) if args.clips_csv else CLIPS_METADATA_PATH

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
    )
    print(f"Checkpoints will be saved to: {SAVE_DIR.resolve()}")
    print(f"Clips metadata source: {CLIPS_CSV}")
    print(f"Workers: {NUM_WORKERS} | Batch size: {BATCH_SIZE} | Mixup: {USE_MIXUP}")

    # ── DataLoaders ────────────────────────────────────────────────────────
    print("Loading datasets...")
    loaders = make_dataloaders(
        clips_csv=CLIPS_CSV,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_mixup=USE_MIXUP,
    )

    from audio.model.dataset import BirdDataset

    train_dataset = loaders["train"].dataset

    assert isinstance(train_dataset, BirdDataset)

    num_classes = train_dataset.num_classes
    class_counts = train_dataset.class_counts
    print(f"Species classes: {num_classes}")

    # ── Model, Loss, Optimizer ─────────────────────────────────────────────
    model = build_model(name=BACKBONE, num_classes=num_classes).to(device)
    criterion = build_loss(loss_type="bce", class_counts=class_counts).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best_val_acc = 0.0
    print("\nStarting Training...\n" + "─" * 60)
    for epoch in range(1, EPOCHS + 1):
        print("epochs:", epoch)
        t0 = time.time()
        train_loss, train_top1, _ = train_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device
        )
        val_loss, val_top1, val_top5 = validate(
            model, loaders["val"], criterion, device
        )
        scheduler.step()
        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_top1 * 100:.1f}% | "
            f"Val Loss: {val_loss:.4f} | Val Top-1: {val_top1 * 100:.1f}% | Val Top-5: {val_top5 * 100:.1f}%"
        )
        # Save best model
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            checkpoint_path = SAVE_DIR / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_top1_acc": val_top1,
                    "val_top5_acc": val_top5,
                    "num_classes": num_classes,
                },
                checkpoint_path,
            )
            print(f"  ✓ New best checkpoint saved to {checkpoint_path}")
    print("─" * 60)
    print(
        f"Training Complete! Best Validation Top-1 Accuracy: {best_val_acc * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
