from __future__ import annotations

import torch.nn as nn

from .labels import NUM_GESTURES


class GestureMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = NUM_GESTURES,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.25,
    ):
        super().__init__()

        h1, h2 = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x):
        return self.network(x)
