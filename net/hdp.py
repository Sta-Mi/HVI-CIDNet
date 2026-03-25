import torch
import torch.nn as nn


class HighDimProjector(nn.Module):
    """Project low-dimensional factors to a high-dimensional latent subspace."""

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=4, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)


class FactorDecoder(nn.Module):
    """Decode a high-dimensional factor back to its native channel space."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x)
