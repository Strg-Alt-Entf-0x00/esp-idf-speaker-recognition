"""
PyTorch Reference Architecture: TDNN-Tiny
==========================================
Repository: esp-idf-speaker-recognition

Pure PyTorch implementation of the TDNN-Tiny acoustic embedding model
for Real-Time Speaker Recognition on ESP32-P4 (512-D embedding).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNNTiny(nn.Module):
    """
    5-layer TDNN + Attentive/Statistical Pooling + 2-layer Fully-Connected projection.
    Input: 80-channel log-mel filterbank features [Batch, 80, Time]
    Output: 512-dimensional L2-normalized speaker embedding [Batch, 512]
    """

    def __init__(self, mel_bins: int = 80, embedding_dim: int = 512):
        super().__init__()
        # 1D Temporal Convolutions (Context frames)
        self.conv1 = nn.Conv1d(mel_bins, 256, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=1)
        self.conv5 = nn.Conv1d(256, 800, kernel_size=1)

        # Dense projection layers
        self.fc1 = nn.Linear(1600, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [Batch, 80, Time] or [80, Time]
        Returns:
            Tensor of shape [Batch, 512]
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)

        # Frame-level feature extraction
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        h5 = F.relu(self.conv5(h4))

        # Temporal Statistics Pooling (mean + std over time frames)
        mean = h5.mean(dim=-1)
        std = torch.sqrt(torch.var(h5, dim=-1, unbiased=False) + 1e-8)
        stats = torch.cat([mean, std], dim=-1)  # [Batch, 1600]

        # Segment-level speaker embedding
        e1 = F.relu(self.fc1(stats))
        e2 = self.fc2(e1)

        # L2-normalization for cosine distance scoring
        return F.normalize(e2, p=2, dim=-1)
