"""Exact distance and optimal moves beyond the table by forward search into it.

Expand the set of states reachable from s in k moves (deduplicated) until one
of them has a known label (the table is exact to max_depth + 1). The first
level k that hits gives d(s) = k + min known distance: any earlier level only
holds states farther than max_depth + 1, so nothing closer is missed. With
max_forward = 6 and a depth-6 table this is exact up to distance 13.

Each frontier state carries the set of first moves of the shortest paths that
reach it, so the optimal first moves of s are the union over the frontier
states at minimum distance.
"""
import torch

from .cube import N_MOVES, STATE_DIM, neighbors
from .table import CHUNK, DistanceTable, state_hash


def _expand_unique(frontier, first, seen_keys):
    """Successors not seen before, deduplicated, with first-move sets OR-merged."""
    out_s, out_k, out_f = [], [], []
    for i in range(0, len(frontier), CHUNK):
        nb = neighbors(frontier[i:i + CHUNK]).view(-1, STATE_DIM)
        f = first[i:i + CHUNK].repeat_interleave(N_MOVES, dim=0)
        h = state_hash(nb)
        new = ~torch.isin(h, seen_keys)
        out_s.append(nb[new]); out_k.append(h[new]); out_f.append(f[new])
    s, k, f = torch.cat(out_s), torch.cat(out_k), torch.cat(out_f)
    uk, inv = torch.unique(k, return_inverse=True)
    merged = torch.zeros(len(uk), N_MOVES, dtype=torch.uint8, device=s.device)
    merged.scatter_reduce_(0, inv.unsqueeze(1).expand(-1, N_MOVES), f, reduce="amax")
    rep = torch.empty(len(uk), dtype=torch.long, device=s.device)
    rep.scatter_(0, inv, torch.arange(len(inv), device=s.device))
    return s[rep], uk, merged


def exact_label(state, table: DistanceTable, max_forward=6):
    """state (40,) or (1,40) -> (distance, optimal-move mask as int).

    Returns (-1, 0) if the distance exceeds max_forward + table.max_depth + 1.
    """
    s0 = state.view(1, STATE_DIM).to(table.device)
    d, m = table.label(s0)
    if d[0] >= 0:
        return int(d[0]), int(m[0])
    # level 1: the 18 successors are distinct, each reached only by its own move
    frontier = neighbors(s0).view(-1, STATE_DIM)
    first = torch.eye(N_MOVES, dtype=torch.uint8, device=table.device)
    seen = torch.sort(torch.cat([state_hash(s0), state_hash(frontier)])).values
    for k in range(1, max_forward + 1):
        d, _ = table.label(frontier)
        known = d >= 0
        if known.any():
            best = d[known].min()
            hit = d == best
            mask_bits = first[hit].amax(dim=0).bool()
            mask = int(sum(1 << i for i in range(N_MOVES) if mask_bits[i]))
            return k + int(best), mask
        if k == max_forward:
            return -1, 0
        frontier, keys, first = _expand_unique(frontier, first, seen)
        seen = torch.sort(torch.cat([seen, keys])).values


def exact_distance(state, table: DistanceTable, max_forward=6):
    return exact_label(state, table, max_forward)[0]
