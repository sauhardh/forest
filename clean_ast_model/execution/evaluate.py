"""
Execution: Benchmark Evaluation with Temporal Mean Pooling.
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from collections import defaultdict
import torch

from config import BATCH_SIZE
from features.mel_pcen import TorchMelPCEN
from features.dataset import build_dataloaders
from model.ast_transformer import ASTBirdClassifier
from model.loss import ClassBalancedBCELoss
from execution.train import compute_metrics


def evaluate(
    checkpoint_path: str | Path,
    clips_csv: str | Path = "outputs/extraction/clips_metadata.csv",
    split: str = "test",
    pool_method: str = "mean",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading '{split}' dataset from: {clips_csv}")
    loaders = build_dataloaders(clips_csv=clips_csv, batch_size=BATCH_SIZE)
    loader = loaders[split]
    dataset = loader.dataset

    model = ASTBirdClassifier(num_classes=dataset.num_classes, dropout=0.0, freeze_layers=0).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mel_transform = TorchMelPCEN().to(device)
    criterion = ClassBalancedBCELoss(class_counts=None).to(device)

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    rec_probs = defaultdict(list)
    rec_targets = {}

    print(f"Evaluating {len(dataset)} clips across {len(dataset.df['recording_id'].unique())} recordings...")

    with torch.no_grad():
        for batch in loader:
            waveforms = batch["waveform"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                specs = mel_transform(waveforms)
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

    n = len(dataset)
    clip_loss = total_loss / n
    clip_top1 = (total_top1 / n) * 100
    clip_top5 = (total_top5 / n) * 100

    rec_top1_correct = 0
    rec_top5_correct = 0
    for rid, prob_list in rec_probs.items():
        stacked = torch.stack(prob_list)
        pooled = stacked.max(dim=0).values if pool_method == "max" else stacked.mean(dim=0)
        target = rec_targets[rid]
        _, top5 = pooled.topk(5)
        if top5[0].item() == target:
            rec_top1_correct += 1
        if target in top5.tolist():
            rec_top5_correct += 1

    num_recs = len(rec_probs)
    rec_top1 = (rec_top1_correct / num_recs) * 100
    rec_top5 = (rec_top5_correct / num_recs) * 100

    print("\n" + "═" * 60)
    print(f"RESULTS on '{split}' set ({n} clips, {num_recs} recordings):")
    print("═" * 60)
    print(f"  Loss:                  {clip_loss:.4f}")
    print(f"  Clip-level Top-1:      {clip_top1:.2f}%")
    print(f"  Clip-level Top-5:      {clip_top5:.2f}%")
    print(f"  Recording-level Top-1: {rec_top1:.2f}% ({rec_top1_correct}/{num_recs} recordings)")
    print(f"  Recording-level Top-5: {rec_top5:.2f}% ({rec_top5_correct}/{num_recs} recordings)")
    print(f"  Pooling Strategy:      {pool_method.upper()}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    pass
