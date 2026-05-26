from __future__ import annotations

import torch
from torch import nn


class SimpleImitationPolicy(nn.Module):
    """Small CNN policy for 3 RGB views plus the first 6 arm joint angles."""

    def __init__(self, image_channels: int = 9, qpos_dim: int = 6, action_dim: int = 7) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + qpos_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )

    def forward(self, images: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(images)
        return self.head(torch.cat((image_features, qpos), dim=1))
