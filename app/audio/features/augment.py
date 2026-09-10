from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import soundfile as sf
from audio import SAMPLE_RATE
from audio.features.mel_transform import MelTransform, N_FFT, HOP_LENGTH


class SpecAugment:
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 40,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        self.F = freq_mask_param
        self.T = time_mask_param
        self.nF = num_freq_masks
        self.nT = num_time_masks

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        spec = spec.copy()
        n_mels, n_frames = spec.shape
        # ── Frequency masking ────────────────────────────────────────────
        # Pick a block of F consecutive Mel channels and zero them out.
        # This prevents the model from keying on a single harmonic band.
        for _ in range(self.nF):
            f = random.randint(0, self.F)  # mask width
            f0 = random.randint(0, max(0, n_mels - f))  # mask start
            spec[f0 : f0 + f, :] = 0.0
        # ── Time masking ─────────────────────────────────────────────────
        # Pick a block of T consecutive frames and zero them out.
        # Forces the model to handle partial / occluded vocalisations.
        for _ in range(self.nT):
            t = random.randint(0, self.T)
            t0 = random.randint(0, max(0, n_frames - t))
            spec[:, t0 : t0 + t] = 0.0
        return spec


import torch
import torch.nn as nn


class TorchSpecAugment(nn.Module):
    """
    GPU-accelerated SpecAugment on 4D batch tensors (B, 1, n_mels, T).
    Runs in < 0.05 ms on CUDA.
    """
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 40,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        super().__init__()
        self.F = freq_mask_param
        self.T = time_mask_param
        self.nF = num_freq_masks
        self.nT = num_time_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        spec = spec.clone()
        _, _, n_mels, n_frames = spec.shape
        for _ in range(self.nF):
            f = random.randint(0, self.F)
            f0 = random.randint(0, max(0, n_mels - f))
            spec[:, :, f0 : f0 + f, :] = 0.0
        for _ in range(self.nT):
            t = random.randint(0, self.T)
            t0 = random.randint(0, max(0, n_frames - t))
            spec[:, :, :, t0 : t0 + t] = 0.0
        return spec


class AcousticMixup:

    """
    Mixup in spectrogram space for multi-label bird audio classification.
    Given two spectrograms X_i, X_j and their one-hot label vectors y_i, y_j:
        X̃ = λ · X_i  +  (1−λ) · X_j
        ỹ = λ · y_i  +  (1−λ) · y_j
    where λ ~ Beta(α, α),  α=0.4 is a good starting value for bioacoustics.
    Rationale: Sound pressure is physically additive, so linearly blending
    two spectrograms is a physically meaningful operation (unlike mixing
    two images of cats). The soft labels reflect that both species are
    simultaneously audible in the mixture.
    Args:
        alpha       : Beta distribution concentration parameter.
        num_classes : Total number of species classes (for one-hot encoding).
    """

    def __init__(self, alpha: float = 0.4, num_classes: int = 304):
        self.alpha = alpha
        self.num_classes = num_classes

    def mix(
        self,
        spec_i: np.ndarray,
        label_i: int,
        spec_j: np.ndarray,
        label_j: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Mix two (spectrogram, class-index) pairs.
        Returns:
            mixed_spec  : (n_mels, T)  float32
            mixed_label : (num_classes,)  float32  – soft label vector
        """
        # Draw mixing coefficient from Beta(α, α)
        lam = np.random.beta(self.alpha, self.alpha)
        # ── Spectrogram blend ────────────────────────────────────────────
        mixed_spec = lam * spec_i + (1.0 - lam) * spec_j
        # ── Soft label blend ─────────────────────────────────────────────
        # One-hot encode both labels, then interpolate.
        y_i = np.zeros(self.num_classes, dtype=np.float32)
        y_j = np.zeros(self.num_classes, dtype=np.float32)
        y_i[label_i] = 1.0
        y_j[label_j] = 1.0
        mixed_label = lam * y_i + (1.0 - lam) * y_j
        return mixed_spec.astype(np.float32), mixed_label


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Background Injection (SNR-controlled)
# ══════════════════════════════════════════════════════════════════════════════
class BackgroundInjector:
    """
    Blends a clean bird waveform with a background noise waveform at a
    controlled Signal-to-Noise Ratio (SNR).
    The final noisy waveform is then passed through MelTransform to
    produce an augmented spectrogram.
    SNR definition used:
        SNR_dB = 20 · log10( RMS(signal) / RMS(noise) )
        → scaling factor α = RMS(signal) / RMS(noise) · 10^(−SNR_dB / 20)
    Args:
        noise_dir   : Directory containing pure background WAV files
                      (wind.wav, rain.wav, forest_bg.wav …).
        snr_range   : (min_dB, max_dB) – SNR is sampled uniformly in this range.
        transform   : MelTransform instance used to produce the final spectrogram.
    """

    def __init__(
        self,
        noise_dir: Path,
        snr_range: tuple[float, float] = (5.0, 20.0),
        transform: MelTransform | None = None,
    ):
        self.noise_dir = noise_dir
        self.snr_range = snr_range
        self.transform = transform or MelTransform()
        # Pre-load all background files (they are short loops)
        self._noise_clips: list[np.ndarray] = []
        for p in sorted(noise_dir.glob("*.wav")):
            audio, sr = sf.read(p, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # stereo → mono
            if sr != SAMPLE_RATE:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            self._noise_clips.append(audio)
        if not self._noise_clips:
            raise FileNotFoundError(f"No background WAV files found in {noise_dir}")

    def _sample_noise_segment(self, n_samples: int) -> np.ndarray:
        """
        Pick a random background clip, loop or trim it to n_samples.
        """
        bg = random.choice(self._noise_clips)
        if len(bg) < n_samples:
            # Loop to fill the required length
            repeats = (n_samples // len(bg)) + 1
            bg = np.tile(bg, repeats)
        # Random crop
        start = random.randint(0, len(bg) - n_samples)
        return bg[start : start + n_samples].copy()

    def inject(self, waveform: np.ndarray) -> np.ndarray:
        """
        Add background noise at a random SNR in self.snr_range.
        Args:
            waveform : Clean bird clip  (n_samples,)  float32 at 32 kHz.
        Returns:
            Augmented Mel spectrogram  (n_mels, T)  float32.
        """
        n_samples = len(waveform)
        noise = self._sample_noise_segment(n_samples)
        # ── SNR-based scaling ────────────────────────────────────────────
        # We want: 20·log10(rms_signal / (α·rms_noise)) = target_snr
        # Solving for α:
        #   α = rms_signal / rms_noise  ×  10^(−target_snr / 20)
        snr_db = random.uniform(*self.snr_range)
        rms_signal = np.sqrt(np.mean(waveform**2)) + 1e-9
        rms_noise = np.sqrt(np.mean(noise**2)) + 1e-9
        alpha = rms_signal / rms_noise * (10.0 ** (-snr_db / 20.0))
        noisy = waveform + alpha * noise
        # Clip to [-1, 1] to prevent float overflow artifacts
        noisy = np.clip(noisy, -1.0, 1.0)
        return self.transform(noisy)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: build a full training-time augmentation pipeline
# ══════════════════════════════════════════════════════════════════════════════
def build_augmentation_pipeline(
    noise_dir: Path | None = None,
    use_pcen: bool = True,
    n_mels: int = 128,
) -> dict:
    """
    Returns a dict of ready-to-use augmentation objects.
    Usage example (inside a PyTorch Dataset.__getitem__):
        aug = build_augmentation_pipeline(noise_dir=Path("data/noise"))
        # 1. Base feature
        spec = aug["transform"](waveform)           # log-Mel or PCEN
        # 2. SpecAugment
        spec = aug["spec_augment"](spec)
        # 3. Mixup  (call in the DataLoader collate_fn for pair-level mixing)
        mixed_spec, mixed_label = aug["mixup"].mix(spec_i, label_i, spec_j, label_j)
        # 4. Background injection (before feature extraction)
        spec = aug["bg_injector"].inject(waveform)
    """
    pipeline: dict = {
        "transform": MelTransform(n_mels=n_mels, use_pcen=use_pcen),
        "spec_augment": SpecAugment(
            freq_mask_param=27,
            time_mask_param=40,
            num_freq_masks=2,
            num_time_masks=2,
        ),
        "mixup": AcousticMixup(alpha=0.4, num_classes=304),
    }
    if noise_dir is not None:
        pipeline["bg_injector"] = BackgroundInjector(
            noise_dir=noise_dir,
            snr_range=(5.0, 20.0),
            transform=pipeline["transform"],
        )
    return pipeline
