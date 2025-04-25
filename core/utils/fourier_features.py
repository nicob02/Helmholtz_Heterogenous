import math
import torch
import torch.nn as nn

class FourierFeatures(nn.Module):
    """
    Projects d‐dim coordinates into a 2*m Fourier feature space:
      [ sin(2π Bᵀx), cos(2π Bᵀx) ]
    where B ∈ R^{d×m} is a fixed random Gaussian matrix.
    """
    def __init__(self, in_dim: int = 2, mapping_size: int = 64, scale: float = 10.0):
        super().__init__()
        # B ~ N(0, scale^2)
        B = torch.randn(in_dim, mapping_size) * scale
        # we register it as a buffer so it moves with .to(device), but is not trained
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_dim]
        # project: [N, mapping_size]
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
