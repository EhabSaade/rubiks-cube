"""Build and freeze all test sets (Step 3). Existing sets are never overwritten;
deep sets resume from a .partial file if interrupted.

    python scripts/build_testsets.py
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubik.cube import apply_sequence, parse_moves, random_states, solved  # noqa: E402
from rubik.exact_search import exact_label  # noqa: E402
from rubik.kociemba_solver import first_move_and_length, solve_states  # noqa: E402
from rubik.paths import DATA_DIR  # noqa: E402
from rubik.scramble import scramble  # noqa: E402
from rubik.table import DistanceTable, state_hash  # noqa: E402
from rubik.testsets import DEEP_DEPTHS, SET_NAMES, TESTSET_DIR  # noqa: E402

N_D7, N_DEEP, N_RANDOM, N_CUBE20 = 2000, 1000, 1000, 10
SUPERFLIP = "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"
# 100 random distance-20 positions from cube20.org (Rokicki, Kociemba, Davidson, Dethridge)
CUBE20_FILE = Path(__file__).resolve().parents[1] / "resources" / "cube20_random100.txt"
WORKERS = int(os.environ.get("RUBIK_WORKERS", 6))


def finish(name, states, dist, dist_exact, mask, mask_known):
    """Add Kociemba labels, save, return manifest entry."""
    states = states.cpu()
    first, length = first_move_and_length(solve_states(states, workers=WORKERS))
    data = {"states": states, "dist": dist.long().cpu(), "dist_exact": dist_exact.cpu(),
            "mask": mask.int().cpu(), "mask_known": mask_known.cpu(),
            "koc_len": torch.tensor(length), "koc_first": torch.tensor(first)}
    assert (data["koc_len"] >= 0).all()
    torch.save(data, TESTSET_DIR / f"{name}.pt")
    print(f"saved {name}: {len(states):,} states", flush=True)


def ones(n):
    return torch.ones(n, dtype=torch.bool)


def from_table(table, name, sel):
    s = table.states[sel]
    finish(name, s, table.dist[sel], ones(len(s)), table.mask[sel], ones(len(s)))


def build_d7(table):
    g = torch.Generator().manual_seed(1007)
    keep, keys = [], set()
    while len(keep) < N_D7:
        s, _ = scramble(4000, 7, g, table.device)
        d, m = table.label(s)
        for row, h, dd, mm in zip(s, state_hash(s).tolist(), d.tolist(), m.tolist()):
            if dd == 7 and h not in keys and len(keep) < N_D7:
                keys.add(h); keep.append((row, mm))
    s = torch.stack([r for r, _ in keep])
    finish("d7", s, torch.full((len(s),), 7), ones(len(s)),
           torch.tensor([m for _, m in keep]), ones(len(s)))


def build_deep(table, d):
    partial = TESTSET_DIR / f"d{d}.partial.pt"
    st = torch.load(partial) if partial.exists() else {"rows": [], "masks": [], "gen_state": None}
    g = torch.Generator().manual_seed(1000 + d)
    if st["gen_state"] is not None:
        g.set_state(st["gen_state"])
    keys = set(state_hash(torch.stack(st["rows"])).tolist()) if st["rows"] else set()
    t0 = time.time()
    while len(st["rows"]) < N_DEEP:
        s, _ = scramble(50, d, g, table.device)
        for row in s:
            h = int(state_hash(row.view(1, -1))[0])
            if h in keys or len(st["rows"]) >= N_DEEP:
                continue
            dd, mm = exact_label(row, table)
            if dd == d:
                keys.add(h); st["rows"].append(row.cpu()); st["masks"].append(mm)
        st["gen_state"] = g.get_state()
        torch.save(st, partial)
        n = len(st["rows"])
        print(f"  d{d}: {n}/{N_DEEP}  ({time.time() - t0:.0f}s)", flush=True)
    s = torch.stack(st["rows"])
    finish(f"d{d}", s, torch.full((len(s),), d), ones(len(s)), torch.tensor(st["masks"]), ones(len(s)))
    partial.unlink()


def build_random():
    s = random_states(N_RANDOM, torch.Generator().manual_seed(2000))
    n = len(s)
    finish("random", s, torch.full((n,), -1), ~ones(n), torch.zeros(n), ~ones(n))


def parse_cube20(line):
    """'F2L2F3...' (face + quarter-turn count) -> move indices."""
    faces = "URFDLB"
    return [faces.index(line[i]) * 3 + int(line[i + 1]) - 1 for i in range(0, len(line), 2)]


def build_d20():
    seqs = [parse_cube20(l.strip()) for l in CUBE20_FILE.read_text().splitlines() if l.strip()]
    seqs = [s for s in seqs if len(s) == 20][:N_CUBE20]
    seqs = [parse_moves(SUPERFLIP)] + seqs
    s = torch.cat([apply_sequence(solved(), q) for q in seqs])
    n = len(s)
    _, length = first_move_and_length(solve_states(s))
    assert min(length) >= 20, "a known distance-20 cube got a Kociemba solution under 20 moves"
    finish("d20_known", s, torch.full((n,), 20), ones(n), torch.zeros(n), ~ones(n))


def main():
    TESTSET_DIR.mkdir(parents=True, exist_ok=True)
    table = DistanceTable.load(DATA_DIR / "table_d6.pt")
    hold = torch.load(DATA_DIR / "holdout_d5_d6.pt")
    builders = {
        "d1_4_all": lambda: from_table(table, "d1_4_all", (table.dist >= 1) & (table.dist <= 4)),
        "holdout_d5": lambda: finish("holdout_d5", *_hold(hold, 5)),
        "holdout_d6": lambda: finish("holdout_d6", *_hold(hold, 6)),
        "d7": lambda: build_d7(table),
        "random": build_random,
        "d20_known": build_d20,
        **{f"d{d}": (lambda d=d: build_deep(table, d)) for d in DEEP_DEPTHS},
    }
    order = ["d1_4_all", "holdout_d5", "holdout_d6", "d7", "random", "d20_known"] +         [f"d{d}" for d in DEEP_DEPTHS]
    for name in order:
        if (TESTSET_DIR / f"{name}.pt").exists():
            print(f"{name}: exists, frozen")
            continue
        builders[name]()
    write_manifest()


def _hold(hold, d):
    sel = hold["dist"] == d
    n = int(sel.sum())
    return hold["states"][sel], hold["dist"][sel], ones(n), hold["mask"][sel], ones(n)


def write_manifest():
    manifest, all_states = {}, []
    for name in SET_NAMES:
        data = torch.load(TESTSET_DIR / f"{name}.pt")
        s = data["states"]
        all_states.append(s)
        manifest[name] = {"n": len(s), "states_sha256": hashlib.sha256(s.numpy().tobytes()).hexdigest(),
                          "mean_koc_len": round(data["koc_len"].float().mean().item(), 3)}
    total = torch.cat(all_states)
    n_unique = len(torch.unique(state_hash(total)))
    manifest["_all"] = {"n": len(total), "unique": n_unique}
    assert n_unique == len(total), "duplicate states across test sets"
    (TESTSET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
