"""
Spectro-temporal feature transforms.

Converts a 1-D waveform x(t) at 32 kHz into a 2-D (Mel × Time) feature map.

Two representations are provided:
  - Log-Mel  : standard log-compressed Mel power spectrogram  (baseline)
  - PCEN     : Per-Channel Energy Normalization spectrogram   (noise-robust)

Window / frame parameters
─────────────────────────
  Sample rate  : 32 000 Hz
  FFT size     : 2 048  (64 ms window)
  Hop size     :   512  (16 ms)  →  188 frames for a 3-second clip
  Mel bands    :   128  (f_min = 500 Hz, f_max = 14 000 Hz)
"""

import numpy as np
import librosa
from audio import SAMPLE_RATE


N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128  # change to 224 for ViT-style patch grids
F_MIN = 500.0  # Hz – below bird vocal range
F_MAX = 14_000.0  # Hz – above most bird fundamentals
WINDOW = "hann"


PCEN_S = 0.025  # IIR smoothing coefficient  (time-constant ≈ 1/s frames)
PCEN_ALPHA = 0.98  # per-channel AGC exponent
PCEN_DELTA = 2.0  # stabilising offset
PCEN_R = 0.5  # root compression exponent
PCEN_EPS = 1e-6  # numerical floor


class MelTransform:
    def __init__(self, n_mels: int = N_MELS, use_pcen: bool = False):
        self.n_mels = n_mels
        self.use_pcen = use_pcen
        # Pre-build the Mel filterbank matrix  (n_mels × (N_FFT//2+1))
        self._mel_fb = librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=self.n_mels,
            fmin=F_MIN,
            fmax=F_MAX,
        )

    def _power_mel(self, waveform: np.ndarray) -> np.ndarray:
        """Return linear-scale Mel power spectrogram  (n_mels × T)."""
        # 1. Short-Time Fourier Transform
        stft = librosa.stft(
            waveform,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=WINDOW,
            center=True,
        )
        # 2. Power spectrum  |STFT|²
        power_spec = np.abs(stft) ** 2  # shape: (N_FFT//2+1, T)
        # 3. Apply Mel filterbank  →  (n_mels, T)
        mel_power = self._mel_fb @ power_spec
        return mel_power.astype(np.float32)

    def log_mel(self, waveform: np.ndarray) -> np.ndarray:
        mel_power = self._power_mel(waveform)
        return np.log(mel_power + PCEN_EPS)

    def pcen(self, waveform: np.ndarray) -> np.ndarray:
        E = self._power_mel(waveform)  # (n_mels, T)
        n_mels, T = E.shape
        # ── Step 1: IIR low-pass smoother  →  M(t, f) ──────────────────
        M = np.zeros_like(E)
        M[:, 0] = PCEN_S * E[:, 0]
        for t in range(1, T):
            M[:, t] = (1.0 - PCEN_S) * M[:, t - 1] + PCEN_S * E[:, t]
        # ── Step 2: Per-channel AGC divisor  (ε + M)^α ─────────────────
        agc_denom = (PCEN_EPS + M) ** PCEN_ALPHA  # (n_mels, T)
        # ── Step 3: Stable root compression ────────────────────────────
        pcen_out = (E / agc_denom + PCEN_DELTA) ** PCEN_R - PCEN_DELTA**PCEN_R
        return pcen_out.astype(np.float32)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        if self.use_pcen:
            return self.pcen(waveform)
        return self.log_mel(waveform)
