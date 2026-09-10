"""
Reads clips_metadata.csv (Phase 3 output), applies Phase 4 transforms,
and serves (spectrogram, label, eco_vector) triples to the training loop.
"""

import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from audio import SAMPLE_RATE, BASE_OUTPUT_DIR
from audio.model import NUM_CLASSES

from audio.features.mel_transform import MelTransform
from audio.features.augment import AcousticMixup, SpecAugment


class BirdDataset:
    def __init__(
        self,
        clips_csv: Path | str,
        split: str,
        use_pcen: bool = True,
        n_mels: int = 128,
        augment: bool = False,
        bg_injector=None,
    ):
        self.split = split
        self.augment = augment
        self.bg_injector = bg_injector

        # Auto-resolve clips_csv path if needed
        csv_p = Path(clips_csv)
        if not csv_p.exists():
            if (BASE_OUTPUT_DIR / csv_p).exists():
                csv_p = BASE_OUTPUT_DIR / csv_p
            elif (BASE_OUTPUT_DIR / "extraction" / csv_p.name).exists():
                csv_p = BASE_OUTPUT_DIR / "extraction" / csv_p.name
            elif (BASE_OUTPUT_DIR.parent / csv_p).exists():
                csv_p = BASE_OUTPUT_DIR.parent / csv_p

        df = pd.read_csv(csv_p)
        self.df = df[df["split"] == split].reset_index(drop=True)

        # -- Build species
        all_species = sorted(df["species"].unique())
        self.species_to_idx: dict[str, int] = {s: i for i, s in enumerate(all_species)}

        self.idx_to_species: dict[int, str] = {
            i: s for s, i in self.species_to_idx.items()
        }
        self.num_classes = len(all_species)

        # -- Per-class sample counts
        self.class_counts = np.zeros(self.num_classes, dtype=np.int64)

        for species, grp in df[df["split"] == "train"].groupby("species"):
            idx = self.species_to_idx[species]
            self.class_counts[idx] = len(grp)

        ## __FEATURE TRANSFORM__
        self.transform = MelTransform(n_mels=n_mels, use_pcen=use_pcen)
        self.spec_augment = (
            SpecAugment(
                freq_mask_param=27,
                time_mask_param=40,
                num_freq_masks=2,
                num_time_masks=2,
            )
            if augment
            else None
        )

    def _resolve_clip_path(self, clip_path: str | Path) -> Path:
        p = Path(clip_path)
        if p.exists():
            return p

        path_str = str(clip_path).replace("\\", "/")
        # Check if path contains extraction/clips
        if "extraction/clips" in path_str:
            sub = path_str[path_str.index("extraction/clips"):]
            candidate = BASE_OUTPUT_DIR / sub
            if candidate.exists():
                return candidate
            if (BASE_OUTPUT_DIR.parent / sub).exists():
                return BASE_OUTPUT_DIR.parent / sub

        # If clip_path starts with "outputs/"
        if path_str.startswith("outputs/"):
            rel_sub = path_str[len("outputs/"):]
            candidate = BASE_OUTPUT_DIR / rel_sub
            if candidate.exists():
                return candidate

        # Check directly under BASE_OUTPUT_DIR
        candidate = BASE_OUTPUT_DIR / p
        if candidate.exists():
            return candidate

        candidate = BASE_OUTPUT_DIR.parent / p
        if candidate.exists():
            return candidate

        return p

    def _load_waveform(self, clip_path: str) -> np.ndarray:
        resolved_path = self._resolve_clip_path(clip_path)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Audio clip not found: {clip_path}\n"
                f"Resolved candidate attempted: {resolved_path}\n"
                f"BASE_OUTPUT_DIR is currently set to: {BASE_OUTPUT_DIR}\n"
                "Tip: Set the FOREST_OUTPUT_DIR environment variable to your data directory, "
                "or mount Google Drive if running in Google Colab."
            )
        audio, _ = sf.read(resolved_path, dtype="float32")
        target = 3 * SAMPLE_RATE

        if len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)))
        elif len(audio) > target:
            audio = audio[:target]

        return audio


    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        waveform = self._load_waveform(row["clip_path"])

        # Background noise injection
        if self.bg_injector is not None and self.augment:
            spec = self.bg_injector.inject(waveform)  # returns spectrogram
        else:
            spec = self.transform(waveform)  # (n_mels, T)

        # Spectogram
        if self.spec_augment is not None:
            spec = self.spec_augment(spec)

        label = self.species_to_idx[row["species"]]
        return {
            "spectrogram": torch.from_numpy(spec).unsqueeze(0),  # (1, n_mels, T)
            "label": torch.tensor(label, dtype=torch.long),
        }


class MixupCollator:
    def __init__(
        self, num_classes: int = NUM_CLASSES, alpha: float = 0.4, p: float = 0.5
    ):
        self.mixup = AcousticMixup(alpha=alpha, num_classes=num_classes)
        self.num_classes = num_classes
        self.alpha = alpha
        self.p = p  # Probability of applying mixup (e.g. 50% of batches or samples)

    def __call__(self, batch: list[dict]) -> dict:
        specs = torch.stack([b["spectrogram"] for b in batch])  # (B, 1, M, T)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)  # (B,)

        one_hot = torch.zeros(len(batch), self.num_classes, dtype=torch.float32)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        if np.random.rand() > self.p:
            return {
                "spectrogram": specs,
                "label": one_hot,
            }

        perm = torch.randperm(len(batch))
        lam = np.random.beta(self.alpha, self.alpha)

        mixed_specs = lam * specs + (1.0 - lam) * specs[perm]
        mixed_labels = lam * one_hot + (1.0 - lam) * one_hot[perm]

        return {
            "spectrogram": mixed_specs,
            "label": mixed_labels,
        }


def make_dataloaders(
    clips_csv: Path,
    use_pcen: bool = True,
    n_mels: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    use_mixup: bool = True,
    bg_injector=None,
) -> dict[str, DataLoader]:
    """
    Returns:
        {
          "train": DataLoader  (with SpecAugment + Mixup + BG injection),
          "val":   DataLoader  (clean),
          "test":  DataLoader  (clean),
        }
    """
    loaders = {}

    for split in ["train", "val", "test"]:
        is_train = split == "train"
        dataset = BirdDataset(
            clips_csv=clips_csv,
            split=split,
            use_pcen=use_pcen,
            n_mels=n_mels,
            augment=is_train,
            bg_injector=bg_injector if is_train else None,
        )

        collate_fn = (
            MixupCollator(num_classes=dataset.num_classes)
            if (is_train and use_mixup)
            else None
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=collate_fn,
            drop_last=is_train,  # keep batch size uniform during training
        )

    return loaders

