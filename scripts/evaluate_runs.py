"""Evaluate trained runs on the frozen test sets (Step 5), after training.

    python scripts/evaluate_runs.py --runs final_mixed_s0 final_kociemba_s0
    python scripts/evaluate_runs.py                 # every final_* run with a finished checkpoint

Metrics (user decision, docs/decisions.md), per run and test set:
- optimal-move rate: top move is optimal, over every cube with known optimal moves
- greedy and beam rollouts (100-move cap, visited states forbidden) on the same
  fixed subsample of at most 1,000 cubes per set as the baselines
  (run_baselines.rollout_indices): solve rate, mean solution length, mean
  excess over the exact distance and over the Kociemba length (solved cubes),
  seconds per cube, and on unsolved cubes the mean misplaced, misoriented and
  wrong (misplaced or misoriented) pieces out of 20 and correct stickers out of 48
- per exact distance as well, for sets that span several distances

Checkpoints (user decision): --checkpoint final (primary) uses the
step-<total_steps> checkpoint and writes results/final_eval/<run>.json;
--checkpoint best_val (secondary) uses the kept checkpoint (every keep_every
steps) with the highest validation target-hit rate, ties to the later step, and
writes results/final_eval/<run>_bestval.json (only a pointer when that is the
final checkpoint). Runs already evaluated are skipped (delete the file to redo one).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from rubik.evaluate import (beam_rollout_batched, correct_stickers, greedy_rollout,  # noqa: E402
                            misoriented_cubies, misplaced_cubies, model_policy, wrong_pieces)
from rubik.model import ResidualMLP  # noqa: E402
from rubik.paths import CHECKPOINT_DIR, RUNS_DIR  # noqa: E402
from rubik.testsets import SET_NAMES, load_set  # noqa: E402
from run_baselines import optimal_move_rate, rollout_indices  # noqa: E402

DEV = "cuda"
OUT_DIR = ROOT / "results" / "final_eval"


def _mean(x):
    return round(x.float().mean().item(), 4) if len(x) else None


def rollout_summary(r, data, idx, seconds):
    solved, length, final = r["solved"].cpu(), r["length"].cpu(), r["final"].cpu()
    dist, exact, koc = data["dist"][idx], data["dist_exact"][idx], data["koc_len"][idx]
    se = solved & exact
    un = ~solved
    out = {"n": len(idx), "solve_rate": _mean(solved),
           "mean_length_solved": _mean(length[solved]),
           "mean_excess_over_exact": _mean((length - dist)[se]),
           "n_solved_with_exact": int(se.sum()),
           "mean_excess_over_kociemba": _mean((length - koc)[solved]),
           "seconds_per_cube": round(seconds / len(idx), 4),
           "n_unsolved": int(un.sum()),
           "unsolved_misplaced": _mean(misplaced_cubies(final[un])),
           "unsolved_misoriented": _mean(misoriented_cubies(final[un])),
           "unsolved_wrong_pieces": _mean(wrong_pieces(final[un])),
           "unsolved_correct_stickers": _mean(correct_stickers(final[un])) if un.any() else None}
    ds = dist[exact].unique().tolist()
    if len(ds) > 1:
        out["by_distance"] = {int(d): {"n": int(((dist == d) & exact).sum()),
                                       "solve_rate": _mean(solved[(dist == d) & exact]),
                                       "mean_length_solved": _mean(length[(dist == d) & exact & solved])}
                              for d in ds}
    return out


def best_val_step(run, cfg):
    """Kept checkpoint with the highest validation target-hit rate (ties: later step)."""
    total, keep = cfg["total_steps"], cfg.get("keep_every", cfg["total_steps"])
    val = {}
    for line in (RUNS_DIR / run / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "val_target_hit" in r and (r["step"] % keep == 0 or r["step"] == total):
            val[r["step"]] = r["val_target_hit"]  # a resumed run may log a step twice: keep the last
    best = max(val, key=lambda s: (val[s], s))
    return best, val[best]


def evaluate_run(run, width, cubes_per_batch, sets, step=None):
    cfg = json.loads((RUNS_DIR / run / "config.json").read_text())
    total = cfg["total_steps"]
    if not (CHECKPOINT_DIR / run / f"step_{total}.pt").exists():
        raise RuntimeError(f"{run} has no step_{total}.pt: not finished")
    ck = torch.load(CHECKPOINT_DIR / run / f"step_{step or total}.pt", map_location=DEV, weights_only=False)
    cfg = ck["config"]
    model = ResidualMLP(cfg["width"], cfg["blocks"]).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    policy = model_policy(model, DEV)
    res = {"run": run, "step": ck["step"], "arm": cfg["arm"], "seed": cfg["seed"],
           "width": cfg["width"], "blocks": cfg["blocks"], "n_stages": cfg["n_stages"],
           "total_steps": total, "beam_width": width, "max_moves": 100, "sets": {}}
    for name in sets:
        data = load_set(name)
        idx = rollout_indices(len(data["states"]), name)
        s = data["states"][idx].to(DEV)
        entry = {"n": len(data["states"]), "optimal_move_rate": optimal_move_rate(policy, data)}
        torch.cuda.synchronize(); t0 = time.time()
        g = greedy_rollout(policy, s)
        torch.cuda.synchronize(); tg = time.time() - t0
        entry["greedy"] = rollout_summary(g, data, idx, tg)
        t0 = time.time()
        b = beam_rollout_batched(policy, s, width=width, cubes_per_batch=cubes_per_batch)
        torch.cuda.synchronize(); tb = time.time() - t0
        entry["beam"] = rollout_summary(b, data, idx, tb)
        res["sets"][name] = entry
        print(f"{run} {name:<11} opt-move {entry['optimal_move_rate']}  "
              f"greedy {entry['greedy']['solve_rate']:.1%}  beam {entry['beam']['solve_rate']:.1%}  "
              f"({tg:.0f}s + {tb:.0f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="run names (default: every finished final_* run)")
    ap.add_argument("--width", type=int, default=4096, help="beam width (Gate C: 4096)")
    ap.add_argument("--cubes_per_batch", type=int, default=32)
    ap.add_argument("--sets", nargs="*", default=list(SET_NAMES))
    ap.add_argument("--checkpoint", choices=["final", "best_val"], default="final")
    args = ap.parse_args()
    runs = args.runs or sorted(p.name for p in CHECKPOINT_DIR.glob("final_*")
                               if (RUNS_DIR / p.name / "config.json").exists())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for run in runs:
        out = OUT_DIR / (f"{run}.json" if args.checkpoint == "final" else f"{run}_bestval.json")
        if out.exists():
            print(f"{run}: already evaluated ({out})")
            continue
        step, val = None, None
        if args.checkpoint == "best_val":
            cfg = json.loads((RUNS_DIR / run / "config.json").read_text())
            if not (CHECKPOINT_DIR / run / f"step_{cfg['total_steps']}.pt").exists():
                raise RuntimeError(f"{run}: not finished")
            step, val = best_val_step(run, cfg)
            print(f"{run}: best validation checkpoint is step {step} (val_target_hit {val:.4f})", flush=True)
            if step == cfg["total_steps"]:
                out.write_text(json.dumps({"run": run, "checkpoint": "best_val", "step": step,
                                           "val_target_hit": val, "same_as_final": f"{run}.json"}, indent=1))
                continue
        t0 = time.time()
        res = evaluate_run(run, args.width, args.cubes_per_batch, args.sets, step)
        res["checkpoint"] = args.checkpoint
        if val is not None:
            res["val_target_hit"] = val
        res["eval_seconds"] = round(time.time() - t0)
        out.write_text(json.dumps(res, indent=1))
        print(f"{run}: done in {res['eval_seconds']}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
