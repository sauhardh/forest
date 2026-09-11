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
from collections import defaultdict


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
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    mel_transform=None,
    spec_augment=None,
    grad_accum: int = 1,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    num_batches = len(loader)

    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        labels = batch["label"].to(device, non_blocking=True)
        if "waveform" in batch and mel_transform is not None:
            waves = batch["waveform"].to(device, non_blocking=True)
            with torch.no_grad():
                specs = mel_transform(waves)
                if spec_augment is not None:
                    specs = spec_augment(specs)
        else:
            specs = batch["spectrogram"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(specs)
            # Scale loss by accumulation steps so effective gradient = full-batch gradient
            loss = criterion(logits, labels) / grad_accum

        scaler.scale(loss).backward()

        # Only step after accumulating grad_accum mini-batches (or at last batch)
        if (i + 1) % grad_accum == 0 or (i + 1) == num_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        metrics = compute_metrics(logits, labels)
        total_loss += loss.item() * grad_accum * len(labels)  # undo the /grad_accum for logging
        total_top1 += metrics["top1_acc"] * len(labels)
        total_top5 += metrics["top5_acc"] * len(labels)

        # ── Print live progress every 50 batches ──
        if (i + 1) % 50 == 0 or (i + 1) == num_batches:
            print(
                f"  Batch [{i + 1:04d}/{num_batches}] | Batch Loss: {loss.item() * grad_accum:.4f}"
            )

    n = len(loader.dataset)
    return total_loss / n, total_top1 / n, total_top5 / n


def validate(
    model,
    loader,
    criterion,
    device,
    mel_transform=None,
    pool_method: str = "mean",
) -> tuple[float, float, float, float, float]:
    """
    Validates model on validation loader.
    Returns:
        (val_loss, clip_top1, clip_top5, rec_top1, rec_top5)
    """
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0

    rec_probs = defaultdict(list)
    rec_targets = {}

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=True)
            if "waveform" in batch and mel_transform is not None:
                waves = batch["waveform"].to(device, non_blocking=True)
                specs = mel_transform(waves)
            else:
                specs = batch["spectrogram"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(specs)
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            metrics = compute_metrics(logits, labels)

            total_loss += loss.item() * len(labels)
            total_top1 += metrics["top1_acc"] * len(labels)
            total_top5 += metrics["top5_acc"] * len(labels)

            # Accumulate probabilities per recording
            if "recording_id" in batch:
                rec_ids = batch["recording_id"]
                for i, rid in enumerate(rec_ids):
                    target_idx = (
                        labels[i].argmax().item()
                        if labels[i].ndim > 0
                        else labels[i].item()
                    )
                    rec_probs[rid].append(probs[i].cpu())
                    rec_targets[rid] = target_idx

    n = len(loader.dataset)
    clip_loss = total_loss / max(n, 1)
    clip_top1 = total_top1 / max(n, 1)
    clip_top5 = total_top5 / max(n, 1)

    # Compute Recording-level Top-1 and Top-5 accuracy
    if rec_probs:
        rec_correct_top1 = 0
        rec_correct_top5 = 0
        for rid, prob_list in rec_probs.items():
            stacked = torch.stack(prob_list)
            if pool_method == "max":
                pooled = stacked.max(dim=0).values
            else:
                pooled = stacked.mean(dim=0)
            target = rec_targets[rid]
            _, top5 = pooled.topk(5)
            if top5[0].item() == target:
                rec_correct_top1 += 1
            if target in top5.tolist():
                rec_correct_top5 += 1

        num_recs = len(rec_probs)
        rec_top1 = rec_correct_top1 / max(num_recs, 1)
        rec_top5 = rec_correct_top5 / max(num_recs, 1)
    else:
        rec_top1 = clip_top1
        rec_top5 = clip_top5

    return clip_loss, clip_top1, clip_top5, rec_top1, rec_top5


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train Bird Sound Classifier")
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-3, help="Weight decay (default: 1e-3)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Classifier dropout rate (default: 0.4)",
    )
    parser.add_argument(
        "--backbone", type=str, default="efficientnet_v2_s", help="Backbone model"
    )
    parser.add_argument(
        "--clips-csv", type=str, default=None, help="Path to clips_metadata.csv"
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Checkpoint save directory"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="DataLoader workers"
    )
    parser.add_argument(
        "--no-mixup", action="store_true", help="Disable acoustic mixup"
    )
    parser.add_argument(
        "--cpu-features",
        action="store_true",
        help="Force feature transforms to run on CPU",
    )
    parser.add_argument(
        "--freq-mask",
        type=int,
        default=36,
        help="SpecAugment frequency mask parameter (default: 36)",
    )
    parser.add_argument(
        "--time-mask",
        type=int,
        default=64,
        help="SpecAugment time mask parameter (default: 64)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience in epochs (0 to disable, default: 5)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.001,
        help="Minimum improvement delta in Val Top-1 for early stopping (default: 0.001)",
    )
    parser.add_argument(
        "--pool-method",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Pooling method for recording-level evaluation: 'mean' or 'max' (default: mean)",
    )
    parser.add_argument(
        "--eval-metric",
        type=str,
        default="rec_top1",
        choices=["rec_top1", "clip_top1"],
        help="Metric to monitor for best checkpoint and early stopping: 'rec_top1' or 'clip_top1' (default: rec_top1)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps. Use with small --batch-size to fit large models in VRAM. "
             "Effective batch = batch_size * grad_accum (default: 1)",
    )
    parser.add_argument(
        "--freeze-ast-layers",
        type=int,
        default=8,
        help="For AST backbone: freeze the first N of 12 encoder layers (default: 8). "
             "Set 0 to fine-tune all layers. Ignored for EfficientNet backbones.",
    )
    return parser


def main(args=None):
    if args is None:
        parser = parse_args()
        # Parse known args so calling in environments like notebooks won't fail
        args, _ = parser.parse_known_args()

    # ── Config ─────────────────────────────────────────────────────────────
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = (
        args.num_workers
        if args.num_workers is not None
        else min(4, (import_os := __import__("os")).cpu_count() or 2)
    )
    EPOCHS = args.epochs
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    DROPOUT = getattr(args, "dropout", 0.4)
    BACKBONE = args.backbone
    USE_MIXUP = not args.no_mixup
    FREQ_MASK = getattr(args, "freq_mask", 36)
    TIME_MASK = getattr(args, "time_mask", 64)
    PATIENCE = getattr(args, "patience", 5)
    MIN_DELTA = getattr(args, "min_delta", 0.001)
    POOL_METHOD = getattr(args, "pool_method", "mean")
    EVAL_METRIC = getattr(args, "eval_metric", "rec_top1")
    GRAD_ACCUM = getattr(args, "grad_accum", 1)
    FREEZE_AST_LAYERS = getattr(args, "freeze_ast_layers", 8)
    SAVE_DIR = Path(args.save_dir) if args.save_dir else CHECKPOINTS_DIR
    CLIPS_CSV = Path(args.clips_csv) if args.clips_csv else CLIPS_METADATA_PATH

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── Feature Acceleration (GPU vs CPU) ──────────────────────────────────
    use_gpu_features = (device.type == "cuda") and (
        not getattr(args, "cpu_features", False)
    )
    if use_gpu_features:
        from audio.features.mel_transform import TorchMelTransform
        from audio.features.augment import TorchSpecAugment

        mel_transform = TorchMelTransform(use_pcen=True).to(device)
        train_spec_augment = TorchSpecAugment(
            freq_mask_param=FREQ_MASK,
            time_mask_param=TIME_MASK,
        ).to(device)
        print(
            f"⚡ GPU Feature Acceleration: ENABLED (Spectrograms & PCEN computed on GPU | SpecAugment F={FREQ_MASK}, T={TIME_MASK})"
        )
    else:
        mel_transform = None
        train_spec_augment = None
        print("Feature transforms: Running on CPU")

    print(
        f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
    )
    print(f"Checkpoints will be saved to: {SAVE_DIR.resolve()}")
    print(f"Clips metadata source: {CLIPS_CSV}")
    print(
        f"Workers: {NUM_WORKERS} | Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM}) | "
        f"Grad accum: {GRAD_ACCUM} | Mixup: {USE_MIXUP} | Dropout: {DROPOUT} | Weight Decay: {WEIGHT_DECAY}"
    )
    print(f"Evaluation: Metric='{EVAL_METRIC}' | Pooling='{POOL_METHOD}'")
    if PATIENCE > 0:
        print(
            f"Early Stopping: ENABLED (Patience: {PATIENCE} epochs, Min Delta: {MIN_DELTA})"
        )
    else:
        print("Early Stopping: DISABLED")

    # ── DataLoaders ────────────────────────────────────────────────────────
    print("Loading datasets...")
    loaders = make_dataloaders(
        clips_csv=CLIPS_CSV,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_mixup=USE_MIXUP,
        return_waveform=use_gpu_features,
    )

    from audio.model.dataset import BirdDataset

    train_dataset = loaders["train"].dataset

    assert isinstance(train_dataset, BirdDataset)

    num_classes = train_dataset.num_classes
    class_counts = train_dataset.class_counts
    print(f"Species classes: {num_classes}")

    # ── Model, Loss, Optimizer ─────────────────────────────────────────────
    model = build_model(
        name=BACKBONE,
        num_classes=num_classes,
        dropout=DROPOUT,
        freeze_layers=FREEZE_AST_LAYERS,
    ).to(device)
    criterion = build_loss(loss_type="bce", class_counts=class_counts).to(device)
    # Only optimize trainable params — frozen layers don't need optimizer states,
    # which saves significant VRAM (Adam keeps 2 extra fp32 copies per param).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print("\nStarting Training...\n" + "─" * 60)
    for epoch in range(1, EPOCHS + 1):
        print("epochs:", epoch)
        t0 = time.time()
        train_loss, train_top1, _ = train_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            mel_transform=mel_transform,
            spec_augment=train_spec_augment,
            grad_accum=GRAD_ACCUM,
        )
        val_loss, clip_top1, clip_top5, rec_top1, rec_top5 = validate(
            model,
            loaders["val"],
            criterion,
            device,
            mel_transform=mel_transform,
            pool_method=POOL_METHOD,
        )
        scheduler.step()
        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_top1 * 100:.1f}% | "
            f"Val Loss: {val_loss:.4f} | Clip Top-1: {clip_top1 * 100:.1f}% | "
            f"Rec Top-1: {rec_top1 * 100:.1f}% (Rec Top-5: {rec_top5 * 100:.1f}%)"
        )
        # Checkpoint & Early stopping check
        current_metric = rec_top1 if EVAL_METRIC == "rec_top1" else clip_top1
        metric_name = "Rec Top-1" if EVAL_METRIC == "rec_top1" else "Clip Top-1"

        if current_metric > (best_val_acc + MIN_DELTA):
            best_val_acc = current_metric
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = SAVE_DIR / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_top1_acc": clip_top1,
                    "val_top5_acc": clip_top5,
                    "rec_top1_acc": rec_top1,
                    "rec_top5_acc": rec_top5,
                    "num_classes": num_classes,
                },
                checkpoint_path,
            )
            print(
                f"  ✓ New best checkpoint saved! ({metric_name}: {current_metric * 100:.2f}% | "
                f"Rec Top-5: {rec_top5 * 100:.1f}% | Clip Top-1: {clip_top1 * 100:.1f}%)"
            )
        else:
            patience_counter += 1
            if PATIENCE > 0:
                print(
                    f"  ℹ No improvement in {metric_name} ({patience_counter}/{PATIENCE} epochs without gain)"
                )
                if patience_counter >= PATIENCE:
                    print(
                        f"\n🛑 Early stopping triggered at epoch {epoch}! "
                        f"No improvement in {metric_name} for {PATIENCE} consecutive epochs. "
                        f"Best checkpoint: {best_val_acc * 100:.2f}% (Epoch {best_epoch})."
                    )
                    break
    print("─" * 60)
    print(
        f"Training Complete! Best Validation {metric_name}: {best_val_acc * 100:.2f}% (from Epoch {best_epoch})"
    )


if __name__ == "__main__":
    main()
