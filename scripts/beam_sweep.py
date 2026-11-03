"""Beam-width sweep (secondary analysis, docs/decisions.md 2026-11-04): the same
beam search and summaries as scripts/evaluate_runs.py at smaller widths, on the
deep test sets, with the baselines' fixed rollout subsamples.

    python scripts/beam_sweep.py --runs final_mixed_s0 final_kociemba_s0
    python scripts/beam_sweep.py --checkpoint best_val --runs final_mixed_s0

Width 4096 is in the main evaluation. More cubes are searched at once at small
widths (cubes_per_batch = max(32, 16384 // width)); that changes speed only,
since cubes never interact. Writes results/beam_sweep/<run>.json or
<run>_bestval.json (a pointer when the best-validation checkpoint is the final one).
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
from evaluate_runs import best_val_step, rollout_summary  # noqa: E402
from rubik.evaluate import beam_rollout_batched, model_policy  # noqa: E402
from rubik.model import ResidualMLP  # noqa: E402
from rubik.paths import CHECKPOINT_DIR, RUNS_DIR  # noqa: E402
from rubik.testsets import load_set  # noqa: E402
from run_baselines import rollout_indices  # noqa: E402

DEV = "cuda"
DEEP_SETS = ["d8", "d9", "d10", "d11", "d12", "d13", "random", "d20_known"]


def sweep_run(run, step, widths, sets):
    ck = torch.load(CHECKPOINT_DIR / run / f"step_{step}.pt", map_location=DEV, weights_only=False)
    cfg = ck["config"]
    model = ResidualMLP(cfg["width"], cfg["blocks"]).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    policy = model_policy(model, DEV)
    res = {"run": run, "step": step, "arm": cfg["arm"], "seed": cfg["seed"], "n_stages": cfg["n_stages"],
           "widths": widths, "max_moves": 100, "sets": {}}
    for name in sets:
        data = load_set(name)
        idx = rollout_indices(len(data["states"]), name)
        s = data["states"][idx].to(DEV)
        res["sets"][name] = {}
        for w in widths:
            torch.cuda.synchronize(); t0 = time.time()
            b = beam_rollout_batched(policy, s, width=w, cubes_per_batch=max(32, 16384 // w))
            torch.cuda.synchronize(); dt = time.time() - t0
            summ = rollout_summary(b, data, idx, dt)
            res["sets"][name][str(w)] = summ
            print(f"{run} step {step} {name:<10} width {w:5d}: solved {summ['solve_rate']:.1%}  "
                  f"length {summ['mean_length_solved']}  ({dt:.0f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--checkpoint", choices=["final", "best_val"], default="final")
    ap.add_argument("--widths", type=int, nargs="+", default=[2, 4, 16, 64, 256, 1024])
    ap.add_argument("--sets", nargs="+", default=DEEP_SETS)
    ap.add_argument("--out_dir", default=str(ROOT / "results" / "beam_sweep"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        out = out_dir / (f"{run}.json" if args.checkpoint == "final" else f"{run}_bestval.json")
        if out.exists():
            print(f"{run}: already swept ({out})")
            continue
        cfg = json.loads((RUNS_DIR / run / "config.json").read_text())
        total = cfg["total_steps"]
        if not (CHECKPOINT_DIR / run / f"step_{total}.pt").exists():
            raise RuntimeError(f"{run}: not finished")
        step = total
        if args.checkpoint == "best_val":
            step, val = best_val_step(run, cfg)
            if step == total:
                out.write_text(json.dumps({"run": run, "checkpoint": "best_val", "step": step,
                                           "val_target_hit": val, "same_as_final": f"{run}.json"}, indent=1))
                print(f"{run}: best validation checkpoint is the final one; pointer written")
                continue
        t0 = time.time()
        res = sweep_run(run, step, args.widths, args.sets)
        res["checkpoint"] = args.checkpoint
        res["seconds"] = round(time.time() - t0)
        out.write_text(json.dumps(res, indent=1))
        print(f"{run}: swept in {res['seconds']}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
