"""Exact distance table built by breadth-first search from the solved cube.

For every state up to max_depth it stores the true half-turn distance and the
set of all optimal moves (an 18-bit mask; bit m set means move m leads to a
state one closer). States one move deeper than the table are labeled exactly
too: a state not in the table with a neighbor at max_depth has distance
max_depth + 1, and its optimal moves are the ones leading into the table.

Lookup sorts states by a 64-bit hash and confirms every hit against the stored
state, so a hash collision can never return a wrong label.
"""
import time

import torch

from .cube import N_MOVES, STATE_DIM, neighbors, solved

# known number of states at each half-turn distance
KNOWN_COUNTS = [1, 18, 243, 3240, 43239, 574908, 7618438, 100803036]

_HASH_WEIGHTS = torch.randint(-2**63, 2**63 - 1, (STATE_DIM,), dtype=torch.int64,
                              generator=torch.Generator().manual_seed(20261102))
CHUNK = 100_000


def state_hash(states):
    """(N, 40) uint8 -> (N,) int64 hash (wrapping linear combination)."""
    w = _HASH_WEIGHTS.to(states.device)
    return (states.long() * w).sum(dim=1)


def _chunks(n, size=CHUNK):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


class DistanceTable:
    def __init__(self, keys, states, dist, mask, max_depth):
        self.keys, self.states, self.dist, self.mask = keys, states, dist, mask
        self.max_depth = max_depth

    @property
    def device(self):
        return self.keys.device

    def __len__(self):
        return len(self.keys)

    # ---------- building ----------
    @classmethod
    def build(cls, max_depth, device="cuda", verbose=True):
        t0 = time.time()
        layers = [solved(1, device)]
        seen = state_hash(layers[0])  # kept sorted
        for d in range(1, max_depth + 1):
            frontier = layers[-1]
            cand_states, cand_keys = [], []
            for sl in _chunks(len(frontier)):
                nb = neighbors(frontier[sl]).view(-1, STATE_DIM)
                h = state_hash(nb)
                new = ~torch.isin(h, seen)
                cand_states.append(nb[new]); cand_keys.append(h[new])
            cs, ck = torch.cat(cand_states), torch.cat(cand_keys)
            ck, order = torch.sort(ck)
            first = torch.ones_like(ck, dtype=torch.bool)
            first[1:] = ck[1:] != ck[:-1]
            layer = cs[order[first]]
            layers.append(layer)
            seen = torch.sort(torch.cat([seen, ck[first]])).values
            if verbose:
                print(f"depth {d}: {len(layer):>10,} states  ({time.time() - t0:.1f}s)")
            del cs, ck, order, cand_states, cand_keys

        states = torch.cat(layers)
        dist = torch.cat([torch.full((len(l),), d, dtype=torch.uint8, device=device)
                          for d, l in enumerate(layers)])
        keys = state_hash(states)
        keys, order = torch.sort(keys)
        states, dist = states[order], dist[order]
        if (keys[1:] == keys[:-1]).any():
            raise RuntimeError("hash collision inside the table; change the hash seed")
        table = cls(keys, states, dist, torch.zeros(len(keys), dtype=torch.int32, device=device),
                    max_depth)
        table.mask = table._optimal_masks(states, dist.long())
        if verbose:
            print(f"optimal-move masks done ({time.time() - t0:.1f}s)")
        return table

    def _optimal_masks(self, states, dist):
        """Bit mask of moves m with dist(s*m) == dist(s) - 1."""
        bits = (1 << torch.arange(N_MOVES, device=states.device)).int()
        out = torch.zeros(len(states), dtype=torch.int32, device=states.device)
        for sl in _chunks(len(states)):
            nd = self.lookup(neighbors(states[sl]).view(-1, STATE_DIM)).view(-1, N_MOVES)
            better = nd == (dist[sl] - 1).unsqueeze(1)
            out[sl] = (better.int() * bits).sum(dim=1).int()
        return out

    # ---------- queries ----------
    def _find(self, states):
        h = state_hash(states)
        idx = torch.searchsorted(self.keys, h).clamp(max=len(self.keys) - 1)
        found = (self.keys[idx] == h) & (self.states[idx] == states).all(dim=1)
        return idx, found

    def lookup(self, states):
        """Distance for states in the table, -1 otherwise. (N,) long."""
        idx, found = self._find(states)
        return torch.where(found, self.dist[idx].long(), torch.full_like(idx, -1))

    def label(self, states):
        """Exact (distance, optimal-move mask) up to max_depth + 1.

        Unknown states (deeper than max_depth + 1) get distance -1 and mask 0.
        """
        out_d = torch.full((len(states),), -1, dtype=torch.long, device=states.device)
        out_m = torch.zeros(len(states), dtype=torch.int32, device=states.device)
        bits = (1 << torch.arange(N_MOVES, device=states.device)).int()
        for sl in _chunks(len(states)):
            s = states[sl]
            idx, found = self._find(s)
            d = torch.where(found, self.dist[idx].long(), torch.full_like(idx, -1))
            m = torch.where(found, self.mask[idx], torch.zeros_like(self.mask[idx]))
            rest = ~found
            if rest.any():
                nd = self.lookup(neighbors(s[rest]).view(-1, STATE_DIM)).view(-1, N_MOVES)
                into = nd == self.max_depth
                deeper = into.any(dim=1)
                d[rest] = torch.where(deeper, self.max_depth + 1, -1)
                m[rest] = (into.int() * bits).sum(dim=1).int()
            out_d[sl], out_m[sl] = d, m
        return out_d, out_m

    # ---------- storage ----------
    def save(self, path):
        torch.save({"keys": self.keys.cpu(), "states": self.states.cpu(), "dist": self.dist.cpu(),
                    "mask": self.mask.cpu(), "max_depth": self.max_depth}, path)

    @classmethod
    def load(cls, path, device="cuda"):
        d = torch.load(path, map_location=device)
        return cls(d["keys"], d["states"], d["dist"], d["mask"], d["max_depth"])


def mask_to_moves(mask):
    """int mask -> list of move indices."""
    return [m for m in range(N_MOVES) if (int(mask) >> m) & 1]
