"""
NNUE (Efficiently Updatable Neural Network) for Quoridor evaluation.

Architecture inspired by Stockfish NNUE:
- Sparse input features → first hidden layer (with clipped ReLU)
- Hidden layers with ReLU
- Single output: position evaluation in centipawns from current player's perspective

The key insight from Stockfish: the first layer is large but sparse,
so it can be incrementally updated when features change (make/unmake moves).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from game import INPUT_SIZE


class NNUE(nn.Module):
    """
    Stockfish-style NNUE for Quoridor.

    Architecture:
        Input (537 sparse features)
        → FC 537 → 256 (ClippedReLU, accumulator — incrementally updatable)
        → FC 256 → 64 (ClippedReLU)
        → FC 64 → 32 (ClippedReLU)
        → FC 32 → 1 (tanh scaled to [-1, 1])

    Output: evaluation from current player's perspective
        +1.0 = winning
        -1.0 = losing
         0.0 = even
    """

    def __init__(self):
        super().__init__()
        # Feature transformer (the "accumulator" in Stockfish terms)
        # This layer is the one that gets incrementally updated
        self.ft = nn.Linear(INPUT_SIZE, 256)

        # Hidden layers
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (batch, INPUT_SIZE) feature tensor
        Returns: (batch, 1) evaluation
        """
        # Feature transformer with clipped ReLU [0, 1]
        x = self.ft(x)
        x = torch.clamp(x, 0.0, 1.0)

        # Hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output: tanh to bound in [-1, 1]
        x = torch.tanh(self.out(x))
        return x


class NNUEAccumulator:
    """
    Efficient accumulator for incremental NNUE updates.

    Instead of recomputing the full first layer on every position,
    we maintain the accumulated output and only add/subtract the
    changed features (like Stockfish does).
    """

    def __init__(self, model: NNUE):
        self.model = model
        self.weight = model.ft.weight.data  # (256, INPUT_SIZE)
        self.bias = model.ft.bias.data      # (256,)
        self.accumulator = None

    def full_refresh(self, features: list) -> torch.Tensor:
        """Full computation of the accumulator from scratch."""
        x = torch.tensor(features, dtype=torch.float32)
        self.accumulator = self.bias.clone()
        # Only add columns where features are non-zero
        for i, val in enumerate(features):
            if val != 0.0:
                self.accumulator += self.weight[:, i] * val
        return self.accumulator

    def update_feature(self, idx: int, old_val: float, new_val: float):
        """Incrementally update accumulator when a single feature changes."""
        if self.accumulator is None:
            raise RuntimeError("Must call full_refresh first")
        delta = new_val - old_val
        if delta != 0.0:
            self.accumulator += self.weight[:, idx] * delta

    def evaluate(self) -> float:
        """Run the rest of the network on the current accumulator."""
        if self.accumulator is None:
            raise RuntimeError("Must call full_refresh first")
        with torch.no_grad():
            x = torch.clamp(self.accumulator, 0.0, 1.0).unsqueeze(0)
            x = F.relu(self.model.fc1(x))
            x = F.relu(self.model.fc2(x))
            x = torch.tanh(self.model.out(x))
            return x.item()


def save_model(model: NNUE, path: str):
    torch.save(model.state_dict(), path)


def load_model(path: str) -> NNUE:
    model = NNUE()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
