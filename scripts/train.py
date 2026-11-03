"""Train one run (Step 4/5). Resumes automatically from the run's last checkpoint.

    python scripts/train.py --config configs/base.yaml --arm true_distance --seed 0
    python scripts/train.py --config configs/base.yaml --arm mixed --seed 0 --set total_steps=2000

Logs to RUNS_DIR/<run>/log.jsonl, checkpoints to CHECKPOINT_DIR/<run>/.
Monitoring uses the pool's validation share only; the frozen test sets are
used only by the final evaluation.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubik.encoding import encode  # noqa: E402
from rubik.model import ResidualMLP, count_parameters  # noqa: E402
from rubik.paths import CHECKPOINT_DIR, RUNS_DIR  # noqa: E402
from rubik.pool import CurriculumSampler, Pool, soft_targets  # noqa: E402

DEV = "cuda"


def load_config(path, arm, seed, overrides):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.update(arm=arm, seed=seed)
    for o in overrides:
        k, v = o.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    cfg.setdefault("run_name", f"{cfg.get('prefix', 'run')}_{arm}_s{seed}")
    return cfg


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, val, n=20_000, seed=123):
    """Optimal-move rate (exact targets only) and target-hit rate on a fixed
    validation sample, reported by true-key bucket."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(val), generator=g)[:n].to(val.states.device)
    s, m, tk, ex = val.states[idx], val.mask[idx].long(), val.true_key[idx].long(), val.exact[idx]
    best = torch.cat([model(encode(s[i:i + 4096])).argmax(1) for i in range(0, len(s), 4096)])
    hit = ((m >> best) & 1).bool()
    out = {"val_target_hit": hit.float().mean().item(),
           "val_opt_exact": hit[ex].float().mean().item() if ex.any() else None}
    for lo, hi in ((1, 4), (5, 7), (8, 9), (10, 12), (13, 99)):
        sel = (tk >= lo) & (tk <= hi)
        if sel.any():
            out[f"val_hit_{lo}_{hi}"] = round(hit[sel].float().mean().item(), 4)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.arm, args.seed, args.set)

    run_dir, ckpt_dir = RUNS_DIR / cfg["run_name"], CHECKPOINT_DIR / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    set_seed(cfg["seed"])
    t0 = time.time()
    train, val = Pool.load().split_validation()
    train, val = train.to(DEV), val.to(DEV)
    print(f"pool: {len(train):,} train, {len(val):,} validation ({time.time() - t0:.0f}s)", flush=True)

    model = ResidualMLP(cfg["width"], cfg["blocks"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["total_steps"]
    warm = cfg.get("warmup_steps", 0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    sampler = CurriculumSampler(train, cfg["arm"], cfg["n_stages"], total, cfg["seed"])
    print(f"model: {count_parameters(model):,} parameters; arm {cfg['arm']}, {cfg['n_stages']} stages "
          f"of {sampler.steps_per_stage} steps", flush=True)

    step = 0
    last = ckpt_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=DEV)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        sampler.load_state_dict(ck["sampler"]); torch.set_rng_state(ck["rng_cpu"]); torch.cuda.set_rng_state(ck["rng_cuda"])
        step = ck["step"]
        print(f"resumed at step {step}", flush=True)

    log = open(run_dir / "log.jsonl", "a")
    bs = cfg["batch_size"]
    t_last, loss_acc, hit_acc, n_acc = time.time(), 0.0, 0.0, 0
    model.train()
    while step < total:
        idx = sampler.sample(step, bs)
        x = encode(train.states[idx])
        y = soft_targets(train.mask[idx])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step(); sched.step()
        step += 1
        loss_acc += loss.item(); n_acc += 1
        hit_acc += ((train.mask[idx].long() >> logits.argmax(1)) & 1).float().mean().item()

        if step % cfg["log_every"] == 0 or step == total:
            rec = {"step": step, "stage": sampler.stage(step - 1), "loss": loss_acc / n_acc,
                   "train_hit": hit_acc / n_acc, "lr": sched.get_last_lr()[0],
                   "steps_per_s": n_acc / (time.time() - t_last)}
            if step % cfg["val_every"] == 0 or step == total:
                rec.update(validate(model, val))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
            t_last, loss_acc, hit_acc, n_acc = time.time(), 0.0, 0.0, 0
        if step % cfg["ckpt_every"] == 0 or step == total:
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                  "sampler": sampler.state_dict(), "rng_cpu": torch.get_rng_state(),
                  "rng_cuda": torch.cuda.get_rng_state(), "step": step, "config": cfg}
            torch.save(ck, ckpt_dir / "last.tmp")
            (ckpt_dir / "last.tmp").replace(last)
            if step % cfg.get("keep_every", 10**12) == 0 or step == total:
                torch.save({"model": model.state_dict(), "step": step, "config": cfg}, ckpt_dir / f"step_{step}.pt")
    log.close()
    print(f"done: {cfg['run_name']} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
