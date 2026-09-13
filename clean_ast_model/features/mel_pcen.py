"""
GPU-accelerated Mel-Spectrogram and Per-Channel Energy Normalization (PCEN).
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

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

        self.register_buffer("window", torch.hann_window(N_FFT))

        mel_fb = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX
        )
        self.register_buffer("mel_fb", torch.from_numpy(mel_fb).float())

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
        power_spec = stft.abs().pow(2)

        # 2. Project onto Mel frequency scale
        mel_power = torch.matmul(self.mel_fb, power_spec)

        # 3. Per-Channel Energy Normalization (PCEN)
        M = torch.empty_like(mel_power)
        M[:, :, 0] = self.s * mel_power[:, :, 0]
        for t in range(1, mel_power.shape[-1]):
            M[:, :, t] = (1.0 - self.s) * M[:, :, t - 1] + self.s * mel_power[:, :, t]

        agc_denom = (self.eps + M).pow(self.alpha)
        spec = (mel_power / agc_denom + self.delta).pow(self.r) - (self.delta**self.r)

        return spec.unsqueeze(1)
