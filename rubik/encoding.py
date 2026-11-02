"""Cubie one-hot encoding: 20 pieces x 24 (position, orientation) = 480 values.

Corner piece j -> one-hot of position*3 + orientation (8 positions x 3).
Edge piece j   -> one-hot of position*2 + orientation (12 positions x 2).
"""
import torch

ENCODING_DIM = 480


def encode(state, dtype=torch.float32):
    """(N, 40) uint8 states -> (N, 480) one-hot tensor on the same device."""
    s = state.long()
    n, dev = s.shape[0], s.device
    cp, co, ep, eo = s[:, 0:8], s[:, 8:16], s[:, 16:28], s[:, 28:40]
    # where each piece is: invert the permutation
    c_pos = torch.argsort(cp, dim=1)
    e_pos = torch.argsort(ep, dim=1)
    c_idx = c_pos * 3 + co.gather(1, c_pos)
    e_idx = e_pos * 2 + eo.gather(1, e_pos)
    offsets = torch.arange(20, device=dev) * 24
    idx = torch.cat([c_idx, e_idx], dim=1) + offsets
    out = torch.zeros(n, ENCODING_DIM, dtype=dtype, device=dev)
    out.scatter_(1, idx, 1)
    return out
