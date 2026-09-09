import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Auto-convert 1D integer labels (from validation/test) to one-hot vectors
        if targets.ndim == 1:
            targets = F.one_hot(targets, num_classes=logits.shape[-1]).float()
        return self.loss_fn(logits, targets)


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Auto-convert 1D integer labels to one-hot vectors
        if targets.ndim == 1:
            targets = F.one_hot(targets, num_classes=logits.shape[-1]).float()

        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        modulating_factor = (1.0 - p_t).pow(self.gamma)
        alpha_weight = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_loss = alpha_weight * modulating_factor * bce
        return focal_loss.mean()


def compute_class_balanced_weights(
    class_counts: np.ndarray, beta: float = 0.999
) -> torch.Tensor:
    """
    Computes Class-Balanced weights (Cui et al., CVPR 2019) using effective numbers:
        E_n = (1 - beta^n) / (1 - beta)
        weight = 1 / E_n
    Normalized so sum(weights) = num_classes.
    """
    counts = np.maximum(class_counts, 1)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_loss(
    loss_type: str = "bce",
    class_counts: np.ndarray | None = None,
    gamma: float = 2.0,
) -> nn.Module:
    """
    Factory function for loss functions.
    """
    loss_type = loss_type.lower()
    if loss_type == "bce":
        pos_weight = None
        if class_counts is not None:
            # Positive weighting for rare species
            weights = compute_class_balanced_weights(class_counts)
            pos_weight = weights
        return SoftTargetBCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type in ("focal", "focal_loss"):
        return MultiLabelFocalLoss(gamma=gamma)
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Use 'bce' or 'focal'.")
