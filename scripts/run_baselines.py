"""Evaluate baselines on the frozen test sets (Step 3).

Optimal-move rate uses every state with known optimal moves. Rollouts use a
fixed subsample of at most ROLLOUT_N states per set (same indices for every
policy and model).

    python scripts/run_baselines.py
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubik.cube import N_MOVES  # noqa: E402
from rubik.evaluate import (greedy_rollout, random_policy, shallow_search_policy,  # noqa: E402
                            wrong_pieces)
from rubik.paths import RUNS_DIR  # noqa: E402
from rubik.testsets import SET_NAMES, load_set  # noqa: E402

ROLLOUT_N = 1000
DEV = "cuda"


def rollout_indices(n, name):
    if n <= ROLLOUT_N:
        return torch.arange(n)
    g = torch.Generator().manual_seed(sum(map(ord, name)))
    return torch.randperm(n, generator=g)[:ROLLOUT_N].sort().values


def optimal_move_rate(policy, data, chunk=5_000):
    known = data["mask_known"]
    if not known.any():
        return None
    s, m = data["states"][known], data["mask"][known].long()
    hits = 0
    for i in range(0, len(s), chunk):
        best = policy(s[i:i + chunk].to(DEV)).argmax(1).cpu()
        hits += int(((m[i:i + chunk] >> best) & 1).sum())
    return hits / len(s)


def evaluate(policy, name, data):
    idx = rollout_indices(len(data["states"]), name)
    s = data["states"][idx].to(DEV)
    t0 = time.time()
    r = greedy_rollout(policy, s)
    dt = time.time() - t0
    solved = r["solved"].cpu()
    out = {"set": name, "n_rollout": len(idx), "solve_rate": solved.float().mean().item(),
           "mean_length_solved": r["length"][r["solved"]].float().mean().item() if solved.any() else None,
           "rollout_seconds": round(dt, 2),
           "unsolved_mean_wrong_pieces": wrong_pieces(r["final"][~r["solved"]]).float().mean().item()
           if (~solved).any() else None,
           "optimal_move_rate": optimal_move_rate(policy, data)}
    return out


def main():
    results = []
    policies = {"random": random_policy(torch.Generator().manual_seed(0)),
                "shallow_search_d2": shallow_search_policy(2)}
    for name in SET_NAMES:
        data = load_set(name)
        koc = {"set": name, "policy": "kociemba", "solve_rate": 1.0,
               "mean_length_solved": data["koc_len"].float().mean().item(),
               "optimal_move_rate": None}
        if data["mask_known"].any():
            k = data["mask_known"] & (data["koc_first"] >= 0)
            koc["optimal_move_rate"] = ((data["mask"][k].long() >> data["koc_first"][k]) & 1).float().mean().item()
        results.append(koc)
        for pname, pol in policies.items():
            r = evaluate(pol, name, data)
            r["policy"] = pname
            results.append(r)
            print(f"{name:<11} {pname:<18} solve {r['solve_rate']:6.1%}  "
                  f"opt-move {r['optimal_move_rate'] if r['optimal_move_rate'] is None else round(r['optimal_move_rate'], 3)}  "
                  f"({r['rollout_seconds']}s)", flush=True)
        print(f"{name:<11} {'kociemba':<18} mean length {koc['mean_length_solved']:.2f}  "
              f"opt-move {koc['optimal_move_rate']}", flush=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "baselines.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
