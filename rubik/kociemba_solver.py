"""Kociemba two-phase solver wrapper: solution, first move and length.

Lengths are upper bounds on true distance. max_depth is the two-phase search
limit: lower values give shorter solutions but take longer, and a limit below
what the cube needs makes the solver search forever, so keep the default.

Parallel solving uses plain subprocesses running rubik.koc_worker, never
multiprocessing: on Windows multiprocessing workers re-import the main script
and with it torch, which reserves 1 to 2 GB per process.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .facelet import to_facelets
from .koc_worker import decode_solution, solve_facelets

DEFAULT_MAX_DEPTH = 24
_REPO = Path(__file__).resolve().parents[1]


def _run_worker(facelets, max_depth):
    env = dict(os.environ, PYTHONPATH=str(_REPO))
    proc = subprocess.run([sys.executable, "-m", "rubik.koc_worker", str(max_depth)],
                          input="\n".join(facelets) + "\n", capture_output=True, text=True,
                          cwd=_REPO, env=env, check=True)
    sols = [decode_solution(l) for l in proc.stdout.splitlines()]
    if len(sols) != len(facelets):
        raise RuntimeError(f"worker returned {len(sols)} solutions for {len(facelets)} cubes")
    return sols


def solve_states(states, max_depth=DEFAULT_MAX_DEPTH, workers=1):
    """(N, 40) states -> list of solutions (lists of move indices or None)."""
    facelets = [to_facelets(row) for row in states.cpu()]
    if workers <= 1 or len(facelets) < 2 * workers:
        return [solve_facelets(f, max_depth) for f in facelets]
    size = -(-len(facelets) // workers)
    parts = [facelets[i:i + size] for i in range(0, len(facelets), size)]
    with ThreadPoolExecutor(len(parts)) as ex:
        results = list(ex.map(lambda p: _run_worker(p, max_depth), parts))
    return [sol for part in results for sol in part]


def first_move_and_length(solutions):
    """Lists of (first move or -1, length or -1)."""
    first = [s[0] if s else -1 for s in solutions]
    length = [len(s) if s is not None else -1 for s in solutions]
    return first, length
