from __future__ import annotations

import torch
from torch import nn


class OidoCTC(nn.Module):
    """Modelo ASR pequeño, entrenable desde cero mediante CTC."""
    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(80, 128, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 192, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(192),
        )
        self.rnn = nn.GRU(192, 192, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
        self.classifier = nn.Linear(384, vocabulary_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features.transpose(1, 2)).transpose(1, 2)
        encoded, _ = self.rnn(encoded)
        return self.classifier(encoded).log_softmax(dim=-1)
