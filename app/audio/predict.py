"""
Run inference on any arbitrary bird audio recording (WAV, MP3, OGG, FLAC, M4A).
Automatically loads, standardizes to 32 kHz, segments into 3-second windows,
computes PCEN log-Mel spectrograms, and outputs Top-K species predictions with confidence scores.
"""

from pathlib import Path
import sys

# Ensure 'app' directory is in sys.path
_current = Path(__file__).resolve()
for _p in [_current.parent, _current.parent.parent]:
    if _p.name == "app" and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
    elif (_p / "app").is_dir() and str(_p / "app") not in sys.path:
        sys.path.insert(0, str(_p / "app"))

import argparse
import numpy as np
import pandas as pd
import torch
import librosa

from audio import SAMPLE_RATE, CHECKPOINTS_DIR, CLIPS_METADATA_PATH, BASE_OUTPUT_DIR
from audio.model.backbone import build_model


def load_species_mapping(clips_csv: Path | str = CLIPS_METADATA_PATH) -> tuple[dict[int, str], dict[str, str]]:
    """
    Builds idx_to_scientific and scientific_to_common dictionaries.
    Order matches BirdDataset: sorted(df['species'].unique()).
    """
    csv_path = Path(clips_csv)
    if not csv_path.exists():
        candidates = [
            BASE_OUTPUT_DIR / "extraction" / "clips_metadata.csv",
            BASE_OUTPUT_DIR / "clips_metadata.csv",
        ]
        for c in candidates:
            if c.exists():
                csv_path = c
                break

    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find clips metadata at {csv_path}")

    df = pd.read_csv(csv_path)
    all_species = sorted(df["species"].unique())
    idx_to_scientific = {i: s for i, s in enumerate(all_species)}

    # Attempt to load English common names
    scientific_to_common = {}
    meta_candidates = [
        BASE_OUTPUT_DIR / "nepal_bird_filtered.csv",
        BASE_OUTPUT_DIR / "nepal_bird_recordings.csv",
    ]
    for meta_p in meta_candidates:
        if meta_p.exists():
            meta_df = pd.read_csv(meta_p)
            if "scientific_name" in meta_df.columns and "english_name" in meta_df.columns:
                mapping = dict(zip(meta_df["scientific_name"], meta_df["english_name"]))
                scientific_to_common.update(mapping)
            break

    return idx_to_scientific, scientific_to_common


def slice_windows(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE, window_sec: float = 3.0, hop_sec: float = 1.5) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Slices waveform into fixed-length windows. Pads with zeros if shorter than window_sec.
    Returns:
        windows: (num_windows, window_samples) float32
        timestamps: list of (start_sec, end_sec)
    """
    win_samples = int(window_sec * sample_rate)
    hop_samples = int(hop_sec * sample_rate)
    total_samples = len(waveform)

    if total_samples <= win_samples:
        padded = np.zeros(win_samples, dtype=np.float32)
        padded[:total_samples] = waveform
        return np.expand_dims(padded, 0), [(0.0, total_samples / sample_rate)]

    windows = []
    timestamps = []
    start = 0
    while start + win_samples <= total_samples:
        windows.append(waveform[start : start + win_samples])
        timestamps.append((start / sample_rate, (start + win_samples) / sample_rate))
        start += hop_samples

    # If the tail end was not covered and has significant content, include the final window
    if start < total_samples and (total_samples - start) >= (win_samples // 2):
        tail = np.zeros(win_samples, dtype=np.float32)
        tail_data = waveform[start:]
        tail[:len(tail_data)] = tail_data
        windows.append(tail)
        timestamps.append((start / sample_rate, total_samples / sample_rate))

    return np.array(windows, dtype=np.float32), timestamps


def predict_audio(
    audio_path: Path | str,
    checkpoint_path: Path | str = CHECKPOINTS_DIR / "best_model.pt",
    clips_csv: Path | str = CLIPS_METADATA_PATH,
    backbone: str = "efficientnet_v2_s",
    top_k: int = 5,
    hop_sec: float = 1.5,
    device: str | torch.device | None = None,
    show_timeline: bool = True,
):
    audio_p = Path(audio_path)
    if not audio_p.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_p.resolve()}")

    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p.resolve()}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"\n📂 Loading audio: {audio_p.name} ({audio_p.resolve()})")
    waveform, sr = librosa.load(str(audio_p), sr=SAMPLE_RATE, mono=True)
    duration_sec = len(waveform) / SAMPLE_RATE
    print(f"⏱  Duration: {duration_sec:.2f}s | Sample Rate: {SAMPLE_RATE} Hz")

    # Slice into 3-second analysis windows
    windows, timestamps = slice_windows(waveform, sample_rate=SAMPLE_RATE, window_sec=3.0, hop_sec=hop_sec)
    print(f"✂  Segmented into {len(windows)} analysis window(s) (window: 3.0s, stride: {hop_sec}s)")

    # Load species names
    idx_to_scientific, scientific_to_common = load_species_mapping(clips_csv)
    num_classes = len(idx_to_scientific)

    # Load Model
    checkpoint = torch.load(ckpt_p, map_location=device, weights_only=False)
    model_num_classes = checkpoint.get("num_classes", num_classes)
    saved_epoch = checkpoint.get("epoch", "unknown")

    model = build_model(name=backbone, num_classes=model_num_classes).to(device)
    # Migrate checkpoint keys if saved with an older version of transformers.
    from audio.model.backbone import ASTAudio
    sd = ASTAudio.migrate_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(sd)
    model.eval()

    # Transforms
    use_gpu_features = (device.type == "cuda")
    if use_gpu_features:
        from audio.features.mel_transform import TorchMelTransform
        mel_transform = TorchMelTransform(use_pcen=True).to(device)
    else:
        from audio.features.mel_transform import MelTransform
        cpu_mel = MelTransform(n_mels=128, use_pcen=True)

    # Inference across windows
    window_tensor = torch.from_numpy(windows).to(device)
    batch_size = 16
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(window_tensor), batch_size):
            batch_waves = window_tensor[i : i + batch_size]
            if use_gpu_features:
                specs = mel_transform(batch_waves)
            else:
                # CPU mel transform per window
                specs_list = [cpu_mel(w.cpu().numpy()) for w in batch_waves]
                specs = torch.from_numpy(np.stack(specs_list)).unsqueeze(1).to(device)

            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(specs)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)  # (num_windows, num_classes)

    # ── Overall Recording Predictions (Mean Pooling) ─────────────────────────
    mean_probs = all_probs.mean(dim=0)
    top_scores, top_indices = mean_probs.topk(top_k)

    print("\n" + "═" * 68)
    print(f"🦅 TOP-{top_k} PREDICTED SPECIES FOR ENTIRE RECORDING")
    print(f"   (Checkpoint from Epoch {saved_epoch})")
    print("═" * 68)
    print(f"{'Rank':<5} | {'Confidence':<10} | {'Scientific Name':<25} | {'English Common Name'}")
    print("─" * 68)

    results = []
    for rank, (score, idx) in enumerate(zip(top_scores, top_indices), start=1):
        sc_name = idx_to_scientific.get(idx.item(), f"Species_{idx.item()}")
        common_name = scientific_to_common.get(sc_name, "—")
        conf_pct = score.item() * 100
        print(f" #{rank:<3} | {conf_pct:>6.2f}%    | {sc_name:<25} | {common_name}")
        results.append({
            "rank": rank,
            "confidence": conf_pct,
            "scientific_name": sc_name,
            "common_name": common_name,
        })
    print("═" * 68)

    # ── Temporal Breakdown Across Windows (if audio > 3s) ───────────────────
    if show_timeline and len(windows) > 1:
        print("\n⏱  TIMELINE BREAKDOWN (Top species per 3-second window):")
        print("─" * 68)
        for w_idx, (t_start, t_end) in enumerate(timestamps):
            w_probs = all_probs[w_idx]
            best_score, best_idx = w_probs.max(dim=0)
            sc_name = idx_to_scientific.get(best_idx.item(), f"Species_{best_idx.item()}")
            common_name = scientific_to_common.get(sc_name, "")
            name_display = f"{sc_name} ({common_name})" if common_name else sc_name
            print(f"  [{t_start:05.1f}s - {t_end:05.1f}s]  {best_score.item() * 100:5.1f}%  →  {name_display}")
        print("─" * 68)

    return results


def main():
    parser = argparse.ArgumentParser(description="Classify any bird sound audio recording")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (WAV, MP3, etc.)")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_DIR / "best_model.pt"), help="Path to checkpoint")
    parser.add_argument("--backbone", type=str, default="efficientnet_v2_s", help="Model backbone used for training (e.g., ast)")
    parser.add_argument("--clips-csv", type=str, default=str(CLIPS_METADATA_PATH), help="Clips metadata for species indexing")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to display")
    parser.add_argument("--hop-sec", type=float, default=1.5, help="Stride between analysis windows in seconds")
    parser.add_argument("--no-timeline", action="store_true", help="Hide window-by-window timeline")
    args = parser.parse_args()

    predict_audio(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,
        clips_csv=args.clips_csv,
        backbone=args.backbone,
        top_k=args.top_k,
        hop_sec=args.hop_sec,
        show_timeline=not args.no_timeline,
    )


if __name__ == "__main__":
    main()
