"""Exact distance beyond the table by forward search into it.

Expand the set of states reachable from s in k moves (deduplicated) until one
of them has a known label (the table is exact to max_depth + 1). The first
level k that hits gives d(s) = k + min known distance: any earlier level only
holds states farther than max_depth + 1, so nothing closer is missed. With
max_forward = 6 and a depth-6 table this is exact up to distance 13.
"""
import torch

from .cube import STATE_DIM, neighbors
from .table import CHUNK, DistanceTable, state_hash


def _expand_unique(frontier, seen_keys):
    out_s, out_k = [], []
    for i in range(0, len(frontier), CHUNK):
        nb = neighbors(frontier[i:i + CHUNK]).view(-1, STATE_DIM)
        h = state_hash(nb)
        new = ~torch.isin(h, seen_keys)
        out_s.append(nb[new]); out_k.append(h[new])
    s, k = torch.cat(out_s), torch.cat(out_k)
    k, order = torch.sort(k)
    first = torch.ones_like(k, dtype=torch.bool)
    first[1:] = k[1:] != k[:-1]
    return s[order[first]], k[first]


def exact_distance(state, table: DistanceTable, max_forward=6):
    """state (40,) or (1,40). Returns exact distance, or -1 if > max_forward + max_depth + 1."""
    frontier = state.view(1, STATE_DIM).to(table.device)
    seen = state_hash(frontier)
    for k in range(max_forward + 1):
        d, _ = table.label(frontier)
        known = d >= 0
        if known.any():
            return k + int(d[known].min())
        if k == max_forward:
            return -1
        frontier, keys = _expand_unique(frontier, seen)
        seen = torch.sort(torch.cat([seen, keys])).values
