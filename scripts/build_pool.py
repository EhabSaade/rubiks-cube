"""Build the depth-8+ training pool (Step 2). Resumable; finished chunks are kept.

Part "path": start cubes (half uniformly random, half scrambles of length 8
to 20) are solved with Kociemba; every cube along each solution is labeled
with the next solution move and the remaining solution length (path
labeling). Cubes within the table (distance <= 7) and cubes from any blocked
test set are dropped. Each cube also gets exact labels up to distance 9
(batched 1- and 2-move lookahead); cubes still unknown are at distance >= 10.

Part "exact": dedicated pools of cubes with exact distance 10, 11 and 12
(10 by batched 3-move lookahead, 11 and 12 by per-cube forward search),
each also labeled by Kociemba.

    python scripts/build_pool.py --part path --target 20000000
    python scripts/build_pool.py --part exact
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubik.batch_label import batch_label  # noqa: E402
from rubik.cube import apply_moves, random_states  # noqa: E402
from rubik.exact_search import exact_label  # noqa: E402
from rubik.kociemba_solver import solve_states  # noqa: E402
from rubik.paths import DATA_DIR  # noqa: E402
from rubik.scramble import scramble  # noqa: E402
from rubik.table import DistanceTable, state_hash  # noqa: E402
from rubik.testsets import blocked_filter  # noqa: E402

POOL_DIR = DATA_DIR / "pool"
DEV = "cuda"
WORKERS = 6
STARTS_PER_CHUNK = 20_000
EXACT_TARGETS = {10: 300_000, 11: 30_000, 12: 10_000}


def start_cubes(n, g):
    """Half uniformly random, half scrambles with lengths spread over 8..20."""
    half = n // 2
    parts = [random_states(half, g, DEV)]
    lengths = torch.randint(8, 21, (n - half,), generator=g)
    for L in lengths.unique().tolist():
        k = int((lengths == L).sum())
        parts.append(scramble(k, L, g, DEV)[0])
    return torch.cat(parts)


def walk_paths(starts, solutions):
    """Every cube along each solution with (next move, remaining length)."""
    lens = torch.tensor([len(s) for s in solutions], device=DEV)
    L = int(lens.max())
    moves = torch.full((len(starts), L), -1, dtype=torch.long, device=DEV)
    for i, s in enumerate(solutions):
        if s:
            moves[i, :len(s)] = torch.tensor(s, device=DEV)
    states, move_out, rem_out = [], [], []
    cur = starts
    for j in range(L):
        live = j < lens
        states.append(cur[live]); move_out.append(moves[live, j]); rem_out.append((lens - j)[live])
        cur = torch.where(live.unsqueeze(1), apply_moves(cur, moves[:, j].clamp(min=0)), cur)
    return torch.cat(states), torch.cat(move_out), torch.cat(rem_out)


def label_to_9(table, states):
    d, m = batch_label(states, table, k=1)
    unknown = d < 0
    if unknown.any():
        d2, m2 = batch_label(states[unknown], table, k=2)
        d[unknown], m[unknown] = d2, m2
    return d, m


def build_path_chunk(table, blocked, idx):
    g = torch.Generator().manual_seed(10_000 + idx)
    starts = start_cubes(STARTS_PER_CHUNK, g)
    sols = solve_states(starts, workers=WORKERS)
    assert all(s is not None for s in sols)
    s, mv, rem = walk_paths(starts, sols)
    d_tab, _ = table.label(s)
    keep = (d_tab < 0) & ~blocked.contains(s)
    s, mv, rem = s[keep], mv[keep], rem[keep]
    # deduplicate within the chunk (first occurrence kept)
    h = state_hash(s)
    hs, order = torch.sort(h, stable=True)
    first = torch.ones_like(hs, dtype=torch.bool)
    first[1:] = hs[1:] != hs[:-1]
    sel = order[first].sort().values
    s, mv, rem = s[sel], mv[sel], rem[sel]
    d, m = label_to_9(table, s)
    return {"states": s.cpu(), "koc_move": mv.cpu(), "koc_len": rem.cpu(),
            "exact_dist": d.cpu(), "exact_mask": m.cpu()}


def run_path(table, blocked, target):
    out = POOL_DIR / "path"
    out.mkdir(parents=True, exist_ok=True)
    done = sorted(out.glob("chunk_*.pt"))
    total = sum(torch.load(p)["states"].shape[0] for p in done)
    idx = len(done)
    t0 = time.time()
    while total < target:
        c = build_path_chunk(table, blocked, idx)
        tmp = out / f"chunk_{idx:04d}.tmp"
        torch.save(c, tmp)
        tmp.rename(out / f"chunk_{idx:04d}.pt")
        total += len(c["states"])
        known = (c["exact_dist"] >= 0).float().mean().item()
        print(f"chunk {idx}: {len(c['states']):,} cubes, exact to 9 for {known:.1%}; "
              f"total {total:,}  ({time.time() - t0:.0f}s)", flush=True)
        idx += 1


def run_exact(table, blocked):
    out = POOL_DIR / "exact"
    out.mkdir(parents=True, exist_ok=True)
    for d, n in EXACT_TARGETS.items():
        final = out / f"d{d}.pt"
        if final.exists():
            print(f"d{d}: exists")
            continue
        partial = out / f"d{d}.partial.pt"
        g = torch.Generator().manual_seed(20_000 + d)
        st = {"states": [], "masks": []}
        if partial.exists():
            saved = torch.load(partial)
            st = {"states": list(saved["states"]), "masks": saved["masks"].tolist()}
            g.set_state(saved["gen"])
        keys = set(state_hash(torch.stack(st["states"])).tolist()) if st["states"] else set()
        t0 = time.time()
        while len(st["states"]) < n:
            batch = 2000 if d == 10 else 50
            s, _ = scramble(batch, d, g, DEV)
            s = s[~blocked.contains(s)]
            if d == 10:
                dd, mm = batch_label(s, table, k=3)
                pairs = [(row, int(m)) for row, x, m in zip(s, dd.tolist(), mm.tolist()) if x == d]
            else:
                pairs = []
                for row in s:
                    x, m = exact_label(row, table)
                    if x == d:
                        pairs.append((row, m))
            for row, m in pairs:
                h = int(state_hash(row.view(1, -1))[0])
                if h not in keys and len(st["states"]) < n:
                    keys.add(h); st["states"].append(row.cpu()); st["masks"].append(m)
            torch.save({"states": torch.stack(st["states"]) if st["states"] else torch.empty(0, 40, dtype=torch.uint8),
                        "masks": torch.tensor(st["masks"]), "gen": g.get_state()}, partial)
            print(f"  d{d}: {len(st['states']):,}/{n:,}  ({time.time() - t0:.0f}s)", flush=True)
        s = torch.stack(st["states"])
        sols = solve_states(s, workers=WORKERS)
        data = {"states": s, "exact_dist": torch.full((len(s),), d), "exact_mask": torch.tensor(st["masks"]).int(),
                "koc_move": torch.tensor([x[0] for x in sols]), "koc_len": torch.tensor([len(x) for x in sols])}
        torch.save(data, final)
        partial.unlink()
        print(f"saved exact d{d}: {len(s):,} cubes", flush=True)


SHALLOW_CHUNK = 500_000
N_D7 = 2_000_000


def _save_shallow_chunk(out, name, s, d, m):
    sols = solve_states(s, workers=WORKERS)
    data = {"states": s.cpu(), "exact_dist": d.long().cpu(), "exact_mask": m.int().cpu(),
            "koc_move": torch.tensor([x[0] if x else -1 for x in sols]),
            "koc_len": torch.tensor([len(x) for x in sols])}
    tmp = out / f"{name}.tmp"
    torch.save(data, tmp)
    tmp.rename(out / f"{name}.pt")


def run_shallow(table, blocked):
    """Depth 1 to 6 table cubes (hold-out excluded) and a depth-7 pool, with
    exact labels and Kociemba lengths, so every arm can order them."""
    out = POOL_DIR / "shallow"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    in_blocked = torch.cat([blocked.contains(table.states[i:i + SHALLOW_CHUNK])
                            for i in range(0, len(table), SHALLOW_CHUNK)])
    sel = (table.dist >= 1) & ~in_blocked
    idx = torch.nonzero(sel).squeeze(1)
    for c, i in enumerate(range(0, len(idx), SHALLOW_CHUNK)):
        name = f"table_{c:03d}"
        if (out / f"{name}.pt").exists():
            continue
        j = idx[i:i + SHALLOW_CHUNK]
        _save_shallow_chunk(out, name, table.states[j], table.dist[j], table.mask[j])
        print(f"{name}: {len(j):,} cubes  ({time.time() - t0:.0f}s)", flush=True)
    g = torch.Generator().manual_seed(30_007)
    n_chunks = N_D7 // SHALLOW_CHUNK
    seen = set()
    for c in range(n_chunks):
        name = f"d7_{c:03d}"
        rows, ms = [], []
        while sum(len(r) for r in rows) < SHALLOW_CHUNK:
            s, _ = scramble(200_000, 7, g, DEV)  # generator advances even for skipped chunks
            d, m = table.label(s)
            keep = (d == 7) & ~blocked.contains(s)
            s, m = s[keep], m[keep]
            h = state_hash(s).tolist()
            new = torch.tensor([x not in seen for x in h], device=DEV)
            seen.update(x for x, n in zip(h, new.tolist()) if n)
            rows.append(s[new]); ms.append(m[new])
        s, m = torch.cat(rows)[:SHALLOW_CHUNK], torch.cat(ms)[:SHALLOW_CHUNK]
        if (out / f"{name}.pt").exists():
            continue
        _save_shallow_chunk(out, name, s, torch.full((len(s),), 7), m)
        print(f"{name}: {len(s):,} cubes  ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["path", "exact", "shallow"], required=True)
    ap.add_argument("--target", type=int, default=20_000_000)
    args = ap.parse_args()
    table = DistanceTable.load(DATA_DIR / "table_d6.pt")
    blocked = blocked_filter(DEV)
    if args.part == "path":
        run_path(table, blocked, args.target)
    elif args.part == "exact":
        run_exact(table, blocked)
    else:
        run_shallow(table, blocked)


if __name__ == "__main__":
    main()
