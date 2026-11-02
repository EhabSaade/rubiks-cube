"""Gate A speed test for the Kociemba two-phase solver.

Measures single-core time per solve and solution length for several max_depth
settings, on uniformly random states and on short scrambles, then multi-core
throughput for the default setting.

    python scripts/speed_kociemba.py
"""
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubik.cube import apply_sequence, is_solved, random_states  # noqa: E402
from rubik.facelet import to_facelets  # noqa: E402
from rubik.kociemba_solver import solve_facelets, solve_states  # noqa: E402
from rubik.scramble import scramble  # noqa: E402


def timed_solve(args):
    f, md = args
    t = time.perf_counter()
    sol = solve_facelets(f, md)
    return time.perf_counter() - t, sol


def bench(name, states, max_depth, per_solve_timeout=20.0):
    times, lengths, timeouts = [], [], 0
    with ProcessPoolExecutor(1) as ex:
        for row in states:
            fut = ex.submit(timed_solve, (to_facelets(row), max_depth))
            try:
                dt, sol = fut.result(timeout=per_solve_timeout)
            except TimeoutError:
                timeouts += 1
                for p in ex._processes.values():
                    p.terminate()
                break
            assert sol is not None and is_solved(apply_sequence(row.unsqueeze(0), sol)).all()
            times.append(dt); lengths.append(len(sol))
    if times:
        print(f"{name:<22} max_depth={max_depth:<3} n={len(times):<4} "
              f"median {statistics.median(times)*1000:8.1f} ms  max {max(times)*1000:8.1f} ms  "
              f"mean length {statistics.mean(lengths):5.2f}  timeouts {timeouts}")
    else:
        print(f"{name:<22} max_depth={max_depth:<3} timed out on first state")


def main():
    g = torch.Generator().manual_seed(0)
    rnd = random_states(100, g)
    for md in (24, 22, 21, 20):
        bench("random states", rnd, md)
    for L in (7, 10, 14):
        s, _ = scramble(100, L, g)
        bench(f"scramble length {L}", s, 24)

    for workers in (1, 4, 8):
        s = random_states(800, g)
        t = time.perf_counter()
        solve_states(s, 24, workers=workers)
        dt = time.perf_counter() - t
        print(f"throughput max_depth=24, {workers} workers: {len(s)/dt:,.0f} solves/s")


if __name__ == "__main__":
    main()
