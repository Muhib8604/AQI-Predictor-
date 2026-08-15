"""
model_definition.py
--------------------
Shared PyTorch architecture. Both training_pipeline.py and predict.py import
AQINet from HERE, so there's a single source of truth for the class — this
is what fixes predict.py's `AQIDeltaNet` ImportError (that class never
existed; the real class is this one).
"""

import torch.nn as nn


class AQINet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)
