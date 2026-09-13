"""
Dataset and DataLoader Pipeline with Acoustic Mixup.
Reads clips_metadata.csv, serves 3.0s raw waveforms to GPU, and applies Mixup on mini-batches.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    SAMPLE_RATE,
    CLIP_SAMPLES,
    NUM_CLASSES,
    MIXUP_ALPHA,
    MIXUP_PROB,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)


class BirdAudioDataset(Dataset):
    """
    Loads 3-second audio clips and their corresponding species labels.
    Returns raw float32 waveforms so feature transforms (Mel/PCEN) run at GPU speed.
    """
    def __init__(self, clips_csv: Path | str, split: str = "train"):
        self.split = split
        csv_path = Path(clips_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path.resolve()}")

        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        # Build consistent alphabetical species-to-index mapping
        all_species = sorted(df["species"].unique())
        self.species_to_idx = {s: i for i, s in enumerate(all_species)}
        self.idx_to_species = {i: s for s, i in self.species_to_idx.items()}
        self.num_classes = len(all_species)

        # Per-class count in training set (used for class-imbalance weighting in loss)
        self.class_counts = np.zeros(self.num_classes, dtype=np.int64)
        for species, grp in df[df["split"] == "train"].groupby("species"):
            idx = self.species_to_idx[species]
            self.class_counts[idx] = len(grp)

    def _load_and_pad(self, audio_path: str) -> np.ndarray:
        """Reads audio file and standardizes length to exactly CLIP_SAMPLES (96,000)."""
        audio, _ = sf.read(audio_path, dtype="float32")
        if len(audio) < CLIP_SAMPLES:
            audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
        elif len(audio) > CLIP_SAMPLES:
            audio = audio[:CLIP_SAMPLES]
        return audio

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        waveform = self._load_and_pad(row["clip_path"])
        label = self.species_to_idx[row["species"]]
        rec_id = str(row.get("recording_id", idx))

        return {
            "waveform": torch.from_numpy(waveform),
            "label": torch.tensor(label, dtype=torch.long),
            "recording_id": rec_id,
        }


class MixupCollator:
    """
    Applies Acoustic Mixup across samples in a training batch.
    Sound pressure waves are physically additive:
        wave_mixed = λ · wave_a + (1 - λ) · wave_b
        label_mixed = λ · one_hot_a + (1 - λ) · one_hot_b
    """
    def __init__(self, num_classes: int = NUM_CLASSES, alpha: float = MIXUP_ALPHA, prob: float = MIXUP_PROB):
        self.num_classes = num_classes
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch: list[dict]) -> dict:
        waveforms = torch.stack([b["waveform"] for b in batch])
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        recording_ids = [b["recording_id"] for b in batch]

        # Convert to one-hot targets
        one_hot = torch.zeros(len(batch), self.num_classes, dtype=torch.float32)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # Decide whether to apply mixup to this batch
        if np.random.rand() > self.prob:
            return {
                "waveform": waveforms,
                "label": one_hot,
                "recording_id": recording_ids,
            }

        # Random permutation for sample pairing
        perm = torch.randperm(len(batch))
        lam = float(np.random.beta(self.alpha, self.alpha))

        mixed_waveforms = lam * waveforms + (1.0 - lam) * waveforms[perm]
        mixed_labels = lam * one_hot + (1.0 - lam) * one_hot[perm]

        return {
            "waveform": mixed_waveforms,
            "label": mixed_labels,
            "recording_id": recording_ids,
        }


def build_dataloaders(clips_csv: Path | str, batch_size: int = BATCH_SIZE) -> dict[str, DataLoader]:
    """Builds Train, Validation, and Test DataLoaders."""
    loaders = {}
    for split in ["train", "val", "test"]:
        is_train = split == "train"
        dataset = BirdAudioDataset(clips_csv=clips_csv, split=split)
        collate_fn = MixupCollator(num_classes=dataset.num_classes) if is_train else None

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY and torch.cuda.is_available(),
            persistent_workers=(NUM_WORKERS > 0),
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
            collate_fn=collate_fn,
            drop_last=is_train,
        )
    return loaders
