"""
GPU-accelerated Feature Extraction: Mel-Spectrogram, PCEN, and SpecAugment.
Transforms raw 32 kHz waveforms into normalized acoustic representations directly in VRAM.
"""

import random
import torch
import torch.nn as nn
import librosa

from config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    F_MIN,
    F_MAX,
    PCEN_S,
    PCEN_ALPHA,
    PCEN_DELTA,
    PCEN_R,
    PCEN_EPS,
    FREQ_MASK_PARAM,
    TIME_MASK_PARAM,
    NUM_FREQ_MASKS,
    NUM_TIME_MASKS,
)


class TorchMelPCEN(nn.Module):
    """
    Computes Mel-Spectrogram and Per-Channel Energy Normalization (PCEN) directly on GPU.
    Replaces Log-Mel compression with adaptive automatic gain control.
    """
    def __init__(self):
        super().__init__()
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH

        # Hann analysis window
        self.register_buffer("window", torch.hann_window(N_FFT))

        # Mel filterbank basis matrix: (n_mels, n_fft // 2 + 1)
        mel_fb = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX
        )
        self.register_buffer("mel_fb", torch.from_numpy(mel_fb).float())

        # PCEN parameters
        self.s = PCEN_S
        self.alpha = PCEN_ALPHA
        self.delta = PCEN_DELTA
        self.r = PCEN_R
        self.eps = PCEN_EPS

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: (Batch, Samples) float32 audio tensor
        Returns:
            spectrogram: (Batch, 1, N_MELS, Time_Frames) PCEN normalized tensor
        """
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)

        # 1. Short-Time Fourier Transform (STFT)
        stft = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        # Power spectrogram: |STFT|^2
        power_spec = stft.abs().pow(2)  # (B, n_fft // 2 + 1, T)

        # 2. Project onto Mel frequency scale
        mel_power = torch.matmul(self.mel_fb, power_spec)  # (B, n_mels, T)

        # 3. Per-Channel Energy Normalization (PCEN)
        # Low-pass filter smoothing across time frames to track background ambient energy
        M = torch.empty_like(mel_power)
        M[:, :, 0] = self.s * mel_power[:, :, 0]
        for t in range(1, mel_power.shape[-1]):
            M[:, :, t] = (1.0 - self.s) * M[:, :, t - 1] + self.s * mel_power[:, :, t]

        # Automatic Gain Control (AGC) divisor
        agc_denom = (self.eps + M).pow(self.alpha)

        # Stable root-compression
        spec = (mel_power / agc_denom + self.delta).pow(self.r) - (self.delta**self.r)

        # Add channel dimension: (B, 1, n_mels, T)
        return spec.unsqueeze(1)


class TorchSpecAugment(nn.Module):
    """
    Applies frequency masking and time masking to spectrogram batches on GPU.
    Forces the attention mechanism to learn distributed acoustic patterns instead
    of overfitting to a single harmonic or transient.
    """
    def __init__(
        self,
        freq_mask_param: int = FREQ_MASK_PARAM,
        time_mask_param: int = TIME_MASK_PARAM,
        num_freq_masks: int = NUM_FREQ_MASKS,
        num_time_masks: int = NUM_TIME_MASKS,
    ):
        super().__init__()
        self.F = freq_mask_param
        self.T = time_mask_param
        self.nF = num_freq_masks
        self.nT = num_time_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: (B, 1, n_mels, n_frames)
        Returns:
            masked_spec: (B, 1, n_mels, n_frames)
        """
        spec = spec.clone()
        _, _, n_mels, n_frames = spec.shape

        # Zero out random horizontal frequency bands
        for _ in range(self.nF):
            f = random.randint(0, self.F)
            f0 = random.randint(0, max(0, n_mels - f))
            spec[:, :, f0 : f0 + f, :] = 0.0

        # Zero out random vertical time blocks
        for _ in range(self.nT):
            t = random.randint(0, self.T)
            t0 = random.randint(0, max(0, n_frames - t))
            spec[:, :, :, t0 : t0 + t] = 0.0

        return spec
