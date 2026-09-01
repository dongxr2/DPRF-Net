"""DPRF-Net model definition used by the paper experiments."""

from __future__ import annotations

import torch
from torch import nn


class DPRFNet(nn.Module):
    """Decoupled preference-reliability fusion with a residual switch."""

    def __init__(
        self,
        vibration_dim: int = 41,
        electrical_dim: int = 55,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        self.vibration_encoder = nn.Sequential(
            nn.Linear(vibration_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.electrical_encoder = nn.Sequential(
            nn.Linear(electrical_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        interaction_dim = hidden_dim * 4
        self.preference_head = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.residual_switch = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, vibration: torch.Tensor, electrical: torch.Tensor):
        hv = self.vibration_encoder(vibration)
        he = self.electrical_encoder(electrical)
        interaction = torch.cat([hv, he, torch.abs(hv - he), hv * he], dim=1)
        preference = torch.softmax(self.preference_head(interaction), dim=1)
        reliability = torch.sigmoid(self.reliability_head(interaction))
        calibrated = preference * reliability.clamp_min(1e-6)
        calibrated = calibrated / calibrated.sum(dim=1, keepdim=True).clamp_min(1e-6)
        adaptive = calibrated[:, :1] * hv + calibrated[:, 1:] * he
        stable = 0.5 * (hv + he)
        alpha = torch.sigmoid(self.residual_switch(interaction))
        fused = stable + alpha * (adaptive - stable)
        return fused, preference, reliability, calibrated, alpha

    def forward(self, vibration: torch.Tensor, electrical: torch.Tensor) -> torch.Tensor:
        fused, *_ = self.encode(vibration, electrical)
        return self.classifier(fused)
