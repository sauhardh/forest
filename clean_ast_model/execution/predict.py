"""
Execution: Single-File Audio Inference for Bird Sound Identification.
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import torch
import librosa

from config import SAMPLE_RATE, CLIP_SAMPLES
from features.mel_pcen import TorchMelPCEN
from model.ast_transformer import ASTBirdClassifier


def slice_windows(waveform: np.ndarray, window_size: int = CLIP_SAMPLES, hop_size: int = CLIP_SAMPLES // 2):
    total_len = len(waveform)
    if total_len <= window_size:
        padded = np.pad(waveform, (0, window_size - total_len))
        return np.expand_dims(padded, 0)

    windows = []
    start = 0
    while start + window_size <= total_len:
        windows.append(waveform[start : start + window_size])
        start += hop_size

    if start < total_len and (total_len - start) >= (window_size // 2):
        tail = np.pad(waveform[start:], (0, window_size - (total_len - start)))
        windows.append(tail)

    return np.array(windows, dtype=np.float32)


def predict_bird(
    audio_path: str | Path,
    checkpoint_path: str | Path,
    species_csv: str | Path = "outputs/extraction/clips_metadata.csv",
    top_k: int = 5,
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(species_csv)
    all_species = sorted(df["species"].unique())
    idx_to_species = {i: s for i, s in enumerate(all_species)}

    model = ASTBirdClassifier(num_classes=len(all_species), dropout=0.0, freeze_layers=0).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mel_transform = TorchMelPCEN().to(device)

    waveform, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    windows = slice_windows(waveform)
    window_tensor = torch.from_numpy(windows).to(device)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            specs = mel_transform(window_tensor)
            logits = model(specs)
            probs = torch.sigmoid(logits)

    mean_probs = probs.mean(dim=0)
    top_scores, top_indices = mean_probs.topk(top_k)

    print("\n" + "═" * 60)
    print(f"🦅 TOP-{top_k} PREDICTED SPECIES: {Path(audio_path).name}")
    print("═" * 60)
    print(f"{'Rank':<5} | {'Confidence':<12} | {'Scientific Name'}")
    print("─" * 60)

    results = []
    for rank, (score, idx) in enumerate(zip(top_scores, top_indices), start=1):
        species_name = idx_to_species.get(idx.item(), f"Species_{idx.item()}")
        conf_pct = score.item() * 100
        print(f" #{rank:<3} | {conf_pct:>6.2f}%      | {species_name}")
        results.append({"rank": rank, "confidence": conf_pct, "species": species_name})
    print("═" * 60 + "\n")

    return results


if __name__ == "__main__":
    pass
