from pathlib import Path
import sys

# Ensure 'app' directory is in sys.path
_current = Path(__file__).resolve()
for _p in [_current.parent, _current.parent.parent, _current.parent.parent.parent]:
    if _p.name == "app" and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
    elif (_p / "app").is_dir() and str(_p / "app") not in sys.path:
        sys.path.insert(0, str(_p / "app"))

import argparse
from collections import defaultdict
import torch

from audio import CLIPS_METADATA_PATH, CHECKPOINTS_DIR
from audio.model.dataset import make_dataloaders
from audio.model.backbone import build_model
from audio.model.loss import build_loss
from audio.model.train import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Bird Sound Classifier Checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "best_model.pt"),
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone", type=str, default="efficientnet_v2_s")
    parser.add_argument("--clips-csv", type=str, default=str(CLIPS_METADATA_PATH))
    parser.add_argument("--cpu-features", action="store_true")
    return parser.parse_args()


def evaluate_model(
    checkpoint_path: Path | str,
    clips_csv: Path | str = CLIPS_METADATA_PATH,
    split: str = "val",
    batch_size: int = 16,
    backbone: str = "efficientnet_v2_s",
    cpu_features: bool = False,
):
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {ckpt_path.resolve()}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 286)
    saved_epoch = checkpoint.get("epoch", "unknown")
    saved_top1 = checkpoint.get("val_top1_acc", None)

    print(f"Checkpoint was saved at Epoch {saved_epoch}")
    if saved_top1 is not None:
        print(f"Recorded Checkpoint Val Top-1: {saved_top1 * 100:.2f}%")

    # ── Feature transforms ──────────────────────────────────────────────────
    use_gpu_features = (device.type == "cuda") and (not cpu_features)
    if use_gpu_features:
        from audio.features.mel_transform import TorchMelTransform
        mel_transform = TorchMelTransform(use_pcen=True).to(device)
    else:
        mel_transform = None

    # ── DataLoaders ────────────────────────────────────────────────────────
    print(f"Loading '{split}' dataset from {clips_csv}...")
    loaders = make_dataloaders(
        clips_csv=clips_csv,
        batch_size=batch_size,
        num_workers=2,
        use_mixup=False,
        return_waveform=use_gpu_features,
    )
    loader = loaders[split]
    dataset = loader.dataset

    # ── Model & Criterion ──────────────────────────────────────────────────
    model = build_model(name=backbone, num_classes=num_classes).to(device)
    # Migrate checkpoint keys if saved with an older version of transformers.
    from audio.model.backbone import ASTAudio
    sd = ASTAudio.migrate_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(sd)
    model.eval()

    criterion = build_loss(loss_type="bce").to(device)

    # ── Evaluation Loop ────────────────────────────────────────────────────
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0

    recording_probs = defaultdict(list)
    recording_targets = {}

    print(f"Evaluating {len(dataset)} clips...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
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

            start_idx = batch_idx * batch_size
            for i, p in enumerate(probs):
                idx = start_idx + i
                if idx < len(dataset.df):
                    row = dataset.df.iloc[idx]
                    rec_id = row.get("recording_id", str(idx))
                    label_idx = labels[i].argmax().item() if labels[i].ndim > 0 else labels[i].item()
                    recording_probs[rec_id].append(p.cpu())
                    recording_targets[rec_id] = label_idx

    n = len(dataset)
    clip_top1 = (total_top1 / n) * 100
    clip_top5 = (total_top5 / n) * 100
    clip_loss = total_loss / n

    print("\n" + "═" * 55)
    print(f"RESULTS on '{split}' set ({n} clips):")
    print(f"  Loss:               {clip_loss:.4f}")
    print(f"  Clip-level Top-1:   {clip_top1:.2f}%")
    print(f"  Clip-level Top-5:   {clip_top5:.2f}%")

    if recording_probs:
        rec_top1_correct = 0
        rec_top5_correct = 0
        for rec_id, prob_list in recording_probs.items():
            mean_prob = torch.stack(prob_list).mean(dim=0)
            target = recording_targets[rec_id]
            _, top5 = mean_prob.topk(5)
            if top5[0].item() == target:
                rec_top1_correct += 1
            if target in top5.tolist():
                rec_top5_correct += 1

        num_recs = len(recording_probs)
        rec_top1 = (rec_top1_correct / num_recs) * 100
        rec_top5 = (rec_top5_correct / num_recs) * 100
        print(f"  Recording-level Top-1: {rec_top1:.2f}% ({rec_top1_correct}/{num_recs} recordings)")
        print(f"  Recording-level Top-5: {rec_top5:.2f}% ({rec_top5_correct}/{num_recs} recordings)")
    print("═" * 55)


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        checkpoint_path=args.checkpoint,
        clips_csv=args.clips_csv,
        split=args.split,
        batch_size=args.batch_size,
        backbone=args.backbone,
        cpu_features=args.cpu_features,
    )
