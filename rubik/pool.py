"""Training pool and curriculum sampler.

All training cubes are precomputed (see scripts/build_pool.py). Each cube has
    target mask : its exact optimal moves where known, otherwise the Kociemba move
    true key    : exact distance where known, otherwise Kociemba length (an
                  upper bound; such cubes are at distance >= 10)
    koc key     : Kociemba length

Arms (decisions log, 2026-11-03):
    true_distance : cumulative stages cut at true-key quantiles
    kociemba      : cumulative stages cut at koc-key quantiles
    mixed         : the whole pool from the first step
Every arm gets the same number of stages and steps; each stage adds the same
share of the pool, so only the order of cubes differs.

A fixed share of the pool (chosen by state hash) is kept out of training as a
validation set for monitoring, so the frozen test sets are not used until
final evaluation.
"""
from pathlib import Path

import torch

from .paths import DATA_DIR
from .table import state_hash

POOL_DIR = DATA_DIR / "pool"
ARMS = ("true_distance", "kociemba", "mixed")
VAL_MOD = 200  # 1 in 200 cubes (0.5%) goes to validation


class Pool:
    def __init__(self, states, mask, true_key, koc_key, exact):
        self.states, self.mask = states, mask.int()
        self.true_key, self.koc_key, self.exact = true_key.short(), koc_key.short(), exact.bool()

    def __len__(self):
        return len(self.states)

    def to(self, device):
        return Pool(self.states.to(device), self.mask.to(device), self.true_key.to(device),
                    self.koc_key.to(device), self.exact.to(device))

    def subset(self, idx):
        return Pool(self.states[idx], self.mask[idx], self.true_key[idx], self.koc_key[idx], self.exact[idx])

    @classmethod
    def from_parts(cls, parts):
        states, mask, tk, kk, ex = [], [], [], [], []
        for p in parts:
            d, m = p["exact_dist"].long(), p["exact_mask"].int()
            known = d >= 0
            koc_bit = (1 << p["koc_move"].clamp(min=0)).int()
            states.append(p["states"])
            mask.append(torch.where(known, m, koc_bit))
            tk.append(torch.where(known, d, p["koc_len"].long()))
            kk.append(p["koc_len"].long())
            ex.append(known)
        return cls(torch.cat(states), torch.cat(mask), torch.cat(tk), torch.cat(kk), torch.cat(ex))

    @classmethod
    def load(cls, pool_dir: Path = POOL_DIR):
        files = (sorted((pool_dir / "shallow").glob("*.pt")) + sorted((pool_dir / "path").glob("chunk_*.pt"))
                 + sorted((pool_dir / "exact").glob("d*[0-9].pt")))
        if not files:
            raise FileNotFoundError(f"no pool files under {pool_dir}")
        return cls.from_parts([torch.load(f) for f in files])

    def split_validation(self):
        """(train, val) by state hash, so the split never depends on load order."""
        is_val = torch.cat([(state_hash(self.states[i:i + 1_000_000]) % VAL_MOD) == 0
                            for i in range(0, len(self), 1_000_000)])
        return self.subset(~is_val), self.subset(is_val)


class CurriculumSampler:
    """Samples batches of pool indices for one arm.

    stage(step) = floor(step / steps_per_stage), capped at n_stages - 1;
    stage s draws uniformly from the first (s + 1) / n_stages of the arm's
    order. Ties in the difficulty key are broken randomly with the seed.
    """

    def __init__(self, pool: Pool, arm, n_stages, total_steps, seed):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm}")
        self.arm, self.n_stages, self.total_steps = arm, n_stages, total_steps
        self.steps_per_stage = -(-total_steps // n_stages)
        dev = pool.states.device
        self.g = torch.Generator(device=dev).manual_seed(seed)
        n = len(pool)
        if arm == "mixed":
            self.order = torch.arange(n, device=dev)
        else:
            key = pool.true_key if arm == "true_distance" else pool.koc_key
            tie = torch.rand(n, generator=self.g, device=dev)
            # sort by key, then by the random tie-breaker
            order = torch.argsort(tie)
            self.order = order[torch.argsort(key[order], stable=True)]
        self.n = n

    def stage(self, step):
        return min(step // self.steps_per_stage, self.n_stages - 1)

    def prefix_len(self, step):
        if self.arm == "mixed":
            return self.n
        return -(-(self.stage(step) + 1) * self.n // self.n_stages)

    def sample(self, step, batch_size):
        k = self.prefix_len(step)
        pos = torch.randint(0, k, (batch_size,), generator=self.g, device=self.order.device)
        return self.order[pos]

    def state_dict(self):
        return {"g": self.g.get_state()}

    def load_state_dict(self, d):
        self.g.set_state(d["g"])


def soft_targets(mask):
    """(N,) int masks -> (N, 18) probabilities spread evenly over the set bits."""
    bits = ((mask.long().unsqueeze(1) >> torch.arange(18, device=mask.device)) & 1).float()
    return bits / bits.sum(1, keepdim=True).clamp(min=1)
