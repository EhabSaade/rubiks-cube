"""Full check of a saved table, plus optimal-move statistics per depth.

    python scripts/check_table.py --depth 6
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubik.cube import N_MOVES, apply_moves  # noqa: E402
from rubik.paths import DATA_DIR  # noqa: E402
from rubik.scramble import scramble  # noqa: E402
from rubik.table import CHUNK, DistanceTable  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=6)
    args = ap.parse_args()
    t = DistanceTable.load(DATA_DIR / f"table_d{args.depth}.pt")
    D = t.max_depth

    # every stored optimal move leads one closer, and no other move does
    bad = 0
    for i in range(0, len(t), CHUNK):
        s, d, m = t.states[i:i + CHUNK], t.dist[i:i + CHUNK].long(), t.mask[i:i + CHUNK].long()
        for mv in range(N_MOVES):
            nd = t.lookup(apply_moves(s, mv))
            bit = ((m >> mv) & 1).bool()
            bad += int((nd[bit] != d[bit] - 1).sum())
            # a missing bit must not be a closer move (neighbor of a depth-D
            # state outside the table is at D+1, never D-1, so this is exact)
            bad += int(((nd == d - 1) & ~bit).sum())
    print("mask errors:", bad)

    n_opt = torch.zeros(len(t), dtype=torch.long, device=t.device)
    for mv in range(N_MOVES):
        n_opt += (t.mask.long() >> mv) & 1
    print("depth  states      mean #optimal moves  share with exactly 1")
    for d in range(1, D + 1):
        sel = n_opt[t.dist == d].float()
        print(f"{d:>5}  {len(sel):>10,}  {sel.mean():>19.2f}  {(sel == 1).float().mean():>19.1%}")

    # labels for random scrambles of length D+1 and D+2
    g = torch.Generator().manual_seed(0)
    for L in (D + 1, D + 2):
        s, _ = scramble(200_000, L, g, t.device)
        d, m = t.label(s)
        hist = torch.bincount(d + 1, minlength=D + 3).tolist()
        print(f"scramble length {L}: unknown={hist[0]:,}, by distance",
              {k: v for k, v in enumerate(hist[1:]) if v})
        known = d >= 1
        assert (m[known] != 0).all()


if __name__ == "__main__":
    main()
