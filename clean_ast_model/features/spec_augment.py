"""
GPU-accelerated SpecAugment (Frequency and Time Masking).
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import random
import torch
import torch.nn as nn

from config import (
    FREQ_MASK_PARAM,
    TIME_MASK_PARAM,
    NUM_FREQ_MASKS,
    NUM_TIME_MASKS,
)


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

        for _ in range(self.nF):
            f = random.randint(0, self.F)
            f0 = random.randint(0, max(0, n_mels - f))
            spec[:, :, f0 : f0 + f, :] = 0.0

        for _ in range(self.nT):
            t = random.randint(0, self.T)
            t0 = random.randint(0, max(0, n_frames - t))
            spec[:, :, :, t0 : t0 + t] = 0.0

        return spec
