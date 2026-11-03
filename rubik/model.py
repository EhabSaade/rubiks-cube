"""Policy network: residual MLP from the 480-value cubie encoding to 18 move logits."""
import torch
from torch import nn

from .cube import N_MOVES
from .encoding import ENCODING_DIM


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.LayerNorm(width), nn.ReLU(),
            nn.Linear(width, width), nn.LayerNorm(width),
        )

    def forward(self, x):
        return torch.relu(x + self.net(x))


class ResidualMLP(nn.Module):
    def __init__(self, width=1024, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(ENCODING_DIM, width), nn.LayerNorm(width), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.head = nn.Linear(width, N_MOVES)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
