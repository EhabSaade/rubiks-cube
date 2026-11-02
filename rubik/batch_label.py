"""Batched exact labels by k-move lookahead into the table.

d(s) = min(table label of s, 1 + min over moves m of d(s*m)), recursively
k levels deep. Exact for distances up to table.max_depth + 1 + k. No
deduplication, so the cost per cube grows 18x per level, but whole batches
run in parallel on the GPU. Practical for k <= 3 (exact to 10 with a depth-6
table).
"""
import torch

from .cube import N_MOVES, STATE_DIM, neighbors
from .table import DistanceTable

INF = 1 << 20


def _dist(states, table, k):
    """(N,) exact distance where <= max_depth+1+k, INF otherwise; also the
    per-move distance via each neighbor (N, 18) or None when k == 0."""
    d0, _ = table.label(states)
    d0 = torch.where(d0 >= 0, d0, torch.full_like(d0, INF))
    if k == 0:
        return d0, None
    sub, _ = _dist(neighbors(states).view(-1, STATE_DIM), table, k - 1)
    via = (sub.view(-1, N_MOVES) + 1).clamp(max=INF)
    return torch.minimum(d0, via.min(1).values), via


def batch_label(states, table: DistanceTable, k=1, chunk=None):
    """(N, 40) -> (dist (N,) long, -1 if unknown; mask (N,) int32, 0 if unknown)."""
    chunk = chunk or max(1, 200_000 // (N_MOVES ** k))
    bits = (1 << torch.arange(N_MOVES, device=states.device)).int()
    out_d = torch.empty(len(states), dtype=torch.long, device=states.device)
    out_m = torch.empty(len(states), dtype=torch.int32, device=states.device)
    for i in range(0, len(states), chunk):
        s = states[i:i + chunk]
        d, via = _dist(s, table, k)
        _, m_tab = table.label(s)
        in_table = m_tab != 0  # labeled directly (also covers the one-deeper ring)
        if via is not None:
            m_via = ((via == d.unsqueeze(1)).int() * bits).sum(1).int()
            m = torch.where(in_table, m_tab, m_via)
        else:
            m = m_tab
        known = d < INF
        out_d[i:i + chunk] = torch.where(known, d, torch.full_like(d, -1))
        out_m[i:i + chunk] = torch.where(known, m, torch.zeros_like(m))
    return out_d, out_m
