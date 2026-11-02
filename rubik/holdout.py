"""Held-out depth 5 and 6 states, blocked from all training.

A fixed share of the table's states at each hold-out depth is drawn with a
fixed seed and saved as a separate file of state hashes and states.
"""
import torch

from .table import DistanceTable, state_hash

HOLDOUT_DEPTHS = (5, 6)
HOLDOUT_SHARE = 0.05
HOLDOUT_SEED = 5


def make_holdout(table: DistanceTable, depths=HOLDOUT_DEPTHS, share=HOLDOUT_SHARE, seed=HOLDOUT_SEED):
    """Returns {'states', 'dist', 'mask'} of the held-out states, sorted by hash."""
    g = torch.Generator().manual_seed(seed)
    picked = []
    for d in depths:
        idx = torch.nonzero(table.dist == d).squeeze(1).cpu()
        k = round(share * len(idx))
        picked.append(idx[torch.randperm(len(idx), generator=g)[:k]])
    idx = torch.cat(picked).sort().values.to(table.device)  # table is sorted by hash
    return {"states": table.states[idx], "dist": table.dist[idx], "mask": table.mask[idx],
            "depths": list(depths), "share": share, "seed": seed}


class HoldoutFilter:
    """Membership test for held-out states, exact (hash plus full-state check)."""

    def __init__(self, states):
        self.keys, order = torch.sort(state_hash(states))
        self.states = states[order]

    @classmethod
    def load(cls, path, device="cuda"):
        return cls(torch.load(path, map_location=device)["states"])

    def contains(self, states):
        if len(self.keys) == 0:
            return torch.zeros(len(states), dtype=torch.bool, device=states.device)
        h = state_hash(states)
        idx = torch.searchsorted(self.keys, h).clamp(max=len(self.keys) - 1)
        return (self.keys[idx] == h) & (self.states[idx] == states).all(dim=1)
