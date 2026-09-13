"""
Class-Balanced Binary Cross-Entropy Loss with Soft Targets.
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedBCELoss(nn.Module):
    """
    Class-Balanced BCE Loss (Cui et al., CVPR 2019).
    Calculates positive loss weights based on the 'effective number' of samples:
        E_n = (1 - β^n) / (1 - β)
        weight = 1 / E_n
    Rare bird species with few training recordings receive proportionally higher
    positive gradients so the model doesn't ignore them.
    """
    def __init__(self, class_counts: np.ndarray | None = None, beta: float = 0.999):
        super().__init__()
        pos_weight = None
        if class_counts is not None:
            counts = np.maximum(class_counts, 1)
            effective_num = 1.0 - np.power(beta, counts)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / np.mean(weights)
            weights = np.clip(weights, 1.0, 25.0)
            pos_weight = torch.tensor(weights, dtype=torch.float32)

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (Batch, Num_Classes) unnormalized model outputs
            targets: (Batch, Num_Classes) soft one-hot targets or (Batch,) integer targets
        """
        if targets.ndim == 1:
            targets = F.one_hot(targets, num_classes=logits.shape[-1]).float()
        return self.loss_fn(logits, targets)
