"""
Execution: 2-Stage Training Loop for the AST Bird Sound Classifier.
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import time
from collections import defaultdict
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    NUM_CLASSES,
    STAGE1_LR,
    STAGE1_FREEZE_LAYERS,
    STAGE1_EPOCHS,
    STAGE2_LR,
    STAGE2_EPOCHS,
    STAGE2_DROPOUT,
    STAGE2_WEIGHT_DECAY,
    BATCH_SIZE,
)
from features.mel_pcen import TorchMelPCEN
from features.spec_augment import TorchSpecAugment
from features.dataset import build_dataloaders
from model.ast_transformer import ASTBirdClassifier
from model.loss import ClassBalancedBCELoss


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """Computes Top-1 and Top-5 accuracy for a batch."""
    with torch.no_grad():
        target_indices = targets.argmax(dim=1) if targets.ndim > 1 else targets
        _, top5 = logits.topk(5, dim=1)
        top1_correct = (top5[:, 0] == target_indices).sum().item()
        top5_correct = (top5 == target_indices.unsqueeze(1)).any(dim=1).sum().item()
        n = len(targets)
        return top1_correct / max(n, 1), top5_correct / max(n, 1)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mel_transform: nn.Module,
    spec_augment: nn.Module,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    num_samples = len(loader.dataset)

    for batch in loader:
        waveforms = batch["waveform"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.no_grad():
            specs = mel_transform(waveforms)
            specs = spec_augment(specs)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(specs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        top1, _ = compute_metrics(logits, labels)
        total_loss += loss.item() * len(labels)
        total_top1 += top1 * len(labels)

    return total_loss / num_samples, total_top1 / num_samples


def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    mel_transform: nn.Module,
) -> tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    rec_probs = defaultdict(list)
    rec_targets = {}
    num_samples = len(loader.dataset)

    with torch.no_grad():
        for batch in loader:
            waveforms = batch["waveform"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            specs = mel_transform(waveforms)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(specs)
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            top1, top5 = compute_metrics(logits, labels)

            total_loss += loss.item() * len(labels)
            total_top1 += top1 * len(labels)
            total_top5 += top5 * len(labels)

            rec_ids = batch["recording_id"]
            for idx, rid in enumerate(rec_ids):
                target_idx = labels[idx].argmax().item() if labels[idx].ndim > 0 else labels[idx].item()
                rec_probs[rid].append(probs[idx].cpu())
                rec_targets[rid] = target_idx

    clip_loss = total_loss / max(num_samples, 1)
    clip_top1 = total_top1 / max(num_samples, 1)
    clip_top5 = total_top5 / max(num_samples, 1)

    rec_top1_correct = 0
    rec_top5_correct = 0
    for rid, prob_list in rec_probs.items():
        pooled = torch.stack(prob_list).mean(dim=0)
        target = rec_targets[rid]
        _, top5 = pooled.topk(5)
        if top5[0].item() == target:
            rec_top1_correct += 1
        if target in top5.tolist():
            rec_top5_correct += 1

    num_recs = len(rec_probs)
    rec_top1 = rec_top1_correct / max(num_recs, 1)
    rec_top5 = rec_top5_correct / max(num_recs, 1)

    return clip_loss, clip_top1, clip_top5, rec_top1, rec_top5


def run_training(
    clips_csv: str = "outputs/extraction/clips_metadata.csv",
    save_dir: str = "checkpoints",
    resume_checkpoint: str | None = None,
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    print(f"🚀 Device: {device} " + (f"({num_gpus} GPUs DataParallel)" if num_gpus > 1 else ""))

    mel_transform = TorchMelPCEN().to(device)
    spec_augment = TorchSpecAugment().to(device)

    print(f"📂 Loading dataset from {clips_csv}...")
    loaders = build_dataloaders(clips_csv=clips_csv, batch_size=BATCH_SIZE)
    class_counts = loaders["train"].dataset.class_counts

    model = ASTBirdClassifier(
        num_classes=NUM_CLASSES,
        dropout=STAGE2_DROPOUT,
        freeze_layers=STAGE1_FREEZE_LAYERS if resume_checkpoint is None else 0,
    ).to(device)

    if num_gpus > 1:
        model = nn.DataParallel(model)

    criterion = ClassBalancedBCELoss(class_counts=class_counts).to(device)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0

    if resume_checkpoint is not None:
        print(f"\n🔄 Resuming for Stage 2 Full Fine-Tuning from: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        raw_model.unfreeze_all_layers()
        best_val_acc = ckpt.get("rec_top1_acc", 0.66)

        learning_rate = STAGE2_LR
        weight_decay = STAGE2_WEIGHT_DECAY
        epochs = STAGE2_EPOCHS
    else:
        learning_rate = STAGE1_LR
        weight_decay = 1e-3
        epochs = STAGE1_EPOCHS

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print(f"Starting Training: {epochs} epochs | LR: {learning_rate} | Weight Decay: {weight_decay}\n" + "─" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, mel_transform, spec_augment
        )
        val_loss, clip_top1, clip_top5, rec_top1, rec_top5 = validate(
            model, loaders["val"], criterion, device, mel_transform
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc*100:.1f}%) | "
            f"Val Loss: {val_loss:.4f} | Clip Top-1: {clip_top1*100:.1f}% | "
            f"Rec Top-1: {rec_top1*100:.1f}% (Rec Top-5: {rec_top5*100:.1f}%)"
        )

        if rec_top1 > best_val_acc:
            best_val_acc = rec_top1
            checkpoint_file = save_path / "best_model.pt"
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "rec_top1_acc": rec_top1,
                    "rec_top5_acc": rec_top5,
                    "clip_top1_acc": clip_top1,
                    "num_classes": NUM_CLASSES,
                },
                checkpoint_file,
            )
            print(f"  ✓ New best checkpoint saved! (Rec Top-1: {rec_top1*100:.2f}%)")

    print("─" * 65)
    print(f"Training Complete! Best Validation Rec Top-1: {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    run_training()
