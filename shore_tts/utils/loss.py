from __future__ import annotations

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F


class FrequencyWeightedMSELoss(nn.Module):
    """MSE loss with linear frequency weighting — higher weight for low frequencies.

    MDCT coefficients are ordered low-to-high along the feature dimension, so we
    apply a linearly decaying weight: w[i] = 1 - alpha * (i / (D - 1)), where
    alpha controls how much high frequencies are down-weighted (0 = uniform, 1 =
    zero weight at highest bin).

    The feature vector is the concatenation of [log_mag (n_bands), norm_spec (n_bins)],
    both ordered low-to-high, so a single weight vector spans the entire last dim.
    """

    def __init__(self, num_channels: int, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        # linear decay: 1.0 at bin 0, (1 - alpha) at the last bin
        weights = 1.0 - alpha * torch.arange(num_channels, dtype=torch.float32) / max(num_channels - 1, 1)
        self.register_buffer("weights", weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        loss = (pred - target).pow(2)
        loss = loss * self.weights  # (..., D)

        if mask is not None:
            loss = loss[mask]
        return loss.mean()
