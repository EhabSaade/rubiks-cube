"""Gate B data check: does Kociemba length track true distance?

For scrambles of length 7..13 computes exact distance (table + forward search)
and Kociemba two-phase length, then prints per-depth summaries and the rank
correlation. Results go to RUNS_DIR/gate_b.json.

    python scripts/gate_b_check.py
"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubik.exact_search import exact_distance  # noqa: E402
from rubik.kociemba_solver import solve_states  # noqa: E402
from rubik.paths import DATA_DIR, RUNS_DIR  # noqa: E402
from rubik.scramble import scramble  # noqa: E402
from rubik.table import DistanceTable  # noqa: E402

PLAN = {7: 200, 8: 200, 9: 200, 10: 200, 11: 200, 12: 200, 13: 60}


def spearman(x, y):
    rx = torch.tensor(x, dtype=torch.float).argsort().argsort().float()
    ry = torch.tensor(y, dtype=torch.float).argsort().argsort().float()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()))


def main():
    table = DistanceTable.load(DATA_DIR / "table_d6.pt")
    g = torch.Generator().manual_seed(42)
    rows = []
    for L, n in PLAN.items():
        t0 = time.time()
        states, _ = scramble(n, L, g, "cuda")
        exact = [exact_distance(s, table) for s in states]
        koc = [len(s) for s in solve_states(states, workers=6)]
        rows += [{"scramble": L, "exact": e, "kociemba": k} for e, k in zip(exact, koc)]
        gap = [k - e for e, k in zip(exact, koc) if e >= 0]
        print(f"scramble {L:>2}: exact mean {statistics.mean(e for e in exact if e >= 0):5.2f} "
              f"(unknown {sum(e < 0 for e in exact)}), kociemba mean {statistics.mean(koc):5.2f}, "
              f"gap mean {statistics.mean(gap):5.2f}, kociemba exact in {sum(x == 0 for x in gap)/len(gap):5.1%}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

    known = [r for r in rows if r["exact"] >= 0]
    rho = spearman([r["exact"] for r in known], [r["kociemba"] for r in known])
    print(f"Spearman rank correlation exact vs kociemba (n={len(known)}): {rho:.3f}")
    by_k = {}
    for r in known:
        by_k.setdefault(r["kociemba"], []).append(r["exact"])
    print("kociemba length -> exact distances seen")
    for k in sorted(by_k):
        print(f"  {k:>2}: n={len(by_k[k]):>4}  exact range {min(by_k[k])}-{max(by_k[k])}  "
              f"mean {statistics.mean(by_k[k]):5.2f}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "gate_b.json").write_text(json.dumps({"rows": rows, "spearman": rho}, indent=1))


if __name__ == "__main__":
    main()
