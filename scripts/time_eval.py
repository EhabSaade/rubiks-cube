"""Clean solve-time measurement for trained runs (user decision, after the
final evaluation, whose seconds-per-cube were measured with GPUs shared).

    CUDA_VISIBLE_DEVICES=0 python scripts/time_eval.py --runs final_mixed_s0 final_kociemba_s0
    CUDA_VISIBLE_DEVICES=0 python scripts/time_eval.py --checkpoint best_val --runs final_mixed_s0

Run one process per GPU and nothing else on that GPU. Per test set it times
greedy and beam rollouts on a fixed subsample (every --stride-th cube of the
baselines' rollout subsample; all cubes of small sets), with the same batching
as scripts/evaluate_runs.py (greedy: the whole subsample at once; beam: 32
cubes at a time), after a warm-up. Seconds per cube = wall time / cubes.
Writes results/final_eval/timing/<run>.json or <run>_bestval.json.
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
from evaluate_runs import best_val_step  # noqa: E402
from rubik.evaluate import beam_rollout_batched, greedy_rollout, model_policy  # noqa: E402
from rubik.model import ResidualMLP  # noqa: E402
from rubik.paths import CHECKPOINT_DIR, RUNS_DIR  # noqa: E402
from rubik.testsets import SET_NAMES, load_set  # noqa: E402
from run_baselines import rollout_indices  # noqa: E402

DEV = "cuda"
OUT_DIR = ROOT / "results" / "final_eval" / "timing"


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    r = fn()
    torch.cuda.synchronize()
    return r, time.time() - t0


def time_run(run, checkpoint, width, cubes_per_batch, stride):
    cfg = json.loads((RUNS_DIR / run / "config.json").read_text())
    step = cfg["total_steps"] if checkpoint == "final" else best_val_step(run, cfg)[0]
    ck = torch.load(CHECKPOINT_DIR / run / f"step_{step}.pt", map_location=DEV, weights_only=False)
    model = ResidualMLP(ck["config"]["width"], ck["config"]["blocks"]).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    policy = model_policy(model, DEV)
    warm = load_set("d8")["states"][:8].to(DEV)
    greedy_rollout(policy, warm)
    beam_rollout_batched(policy, warm, width=width, cubes_per_batch=cubes_per_batch)
    res = {"run": run, "checkpoint": checkpoint, "step": step, "beam_width": width,
           "cubes_per_batch": cubes_per_batch, "stride": stride,
           "gpu": torch.cuda.get_device_name(0), "sets": {}}
    for name in SET_NAMES:
        data = load_set(name)
        idx = rollout_indices(len(data["states"]), name)
        idx = idx[::stride] if len(idx) > 100 else idx
        s = data["states"][idx].to(DEV)
        g, tg = timed(lambda: greedy_rollout(policy, s))
        b, tb = timed(lambda: beam_rollout_batched(policy, s, width=width, cubes_per_batch=cubes_per_batch))
        res["sets"][name] = {"n": len(idx),
                             "greedy_seconds_per_cube": round(tg / len(idx), 5),
                             "beam_seconds_per_cube": round(tb / len(idx), 5),
                             "greedy_solve_rate": round(g["solved"].float().mean().item(), 4),
                             "beam_solve_rate": round(b["solved"].float().mean().item(), 4)}
        print(f"{run} step {step} {name:<11} n={len(idx):4d}  greedy {tg / len(idx):.4f} s/cube  "
              f"beam {tb / len(idx):.3f} s/cube", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--checkpoint", choices=["final", "best_val"], default="final")
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--cubes_per_batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=5, help="every n-th cube of the rollout subsample")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        out = OUT_DIR / (f"{run}.json" if args.checkpoint == "final" else f"{run}_bestval.json")
        if out.exists():
            print(f"{run}: already timed ({out})")
            continue
        if args.checkpoint == "best_val":
            cfg = json.loads((RUNS_DIR / run / "config.json").read_text())
            if best_val_step(run, cfg)[0] == cfg["total_steps"]:
                print(f"{run}: best validation checkpoint is the final one; not timed again")
                continue
        t0 = time.time()
        res = time_run(run, args.checkpoint, args.width, args.cubes_per_batch, args.stride)
        res["seconds"] = round(time.time() - t0)
        out.write_text(json.dumps(res, indent=1))
        print(f"{run}: timed in {res['seconds']}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
