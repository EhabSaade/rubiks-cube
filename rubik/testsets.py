"""Frozen test sets: loading and the isolation filter used by training.

Each set is DATA_DIR/testsets/<name>.pt holding
    states     (N, 40) uint8
    dist       (N,) long, true distance, -1 if unknown
    dist_exact (N,) bool
    mask       (N,) int32, all optimal moves, 0 if unknown
    mask_known (N,) bool
    koc_len    (N,) long, Kociemba solution length
    koc_first  (N,) long, Kociemba first move (-1 for solved)

d1_4_all is in-distribution (every state of depth 1 to 4, which training also
sees); every other set is blocked from training.
"""
import torch

from .holdout import HoldoutFilter
from .paths import DATA_DIR

TESTSET_DIR = DATA_DIR / "testsets"
DEEP_DEPTHS = range(8, 14)
SET_NAMES = (["d1_4_all", "holdout_d5", "holdout_d6", "d7"]
             + [f"d{d}" for d in DEEP_DEPTHS] + ["random", "d20_known"])
IN_DISTRIBUTION = {"d1_4_all"}


def load_set(name, device="cpu"):
    return torch.load(TESTSET_DIR / f"{name}.pt", map_location=device)


def blocked_filter(device="cuda"):
    """Membership filter for every state training must never see."""
    states = [load_set(n, device)["states"] for n in SET_NAMES if n not in IN_DISTRIBUTION]
    return HoldoutFilter(torch.cat(states))
