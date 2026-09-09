from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


try:
    from audio.model.dataset import make_dataloaders
    from audio.model.backbone import build_model
    from audio.model.loss import build_loss
    from audio import CLIPS_METADATA_PATH
except ImportError:
    from model.dataset import make_dataloaders
    from model.backbone import build_model
    from model.loss import build_loss
    from audio import CLIPS_METADATA_PATH


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


def main():
    # ── Config ─────────────────────────────────────────────────────────────
    BATCH_SIZE = 16  # 16 fits safely inside 4GB VRAM
    NUM_WORKERS = 4  # Optimal for Ryzen 5
    EPOCHS = 25
    LR = 5e-4
    WEIGHT_DECAY = 1e-4
    BACKBONE = "efficientnet_v2_s"
    SAVE_DIR = Path("outputs/checkpoints")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
    )
    # ── DataLoaders ────────────────────────────────────────────────────────
    print("Loading datasets...")
    loaders = make_dataloaders(
        clips_csv=CLIPS_METADATA_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_mixup=True,
    )

    from audio.model.dataset import BirdDataset

    train_dataset = loaders["train"].dataset

    assert isinstance(train_dataset, BirdDataset)

    num_classes = train_dataset.num_classes
    class_counts = train_dataset.class_counts
    print(f"Species classes: {num_classes}")

    # num_classes = loaders["train"].dataset.num_classes
    # class_counts = loaders["train"].dataset.class_counts
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
