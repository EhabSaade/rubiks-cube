"""Kociemba two-phase solver wrapper: solution, first move and length.

Lengths are upper bounds on true distance. max_depth is the two-phase search
limit: lower values give shorter solutions but take longer, and a limit below
what the cube needs makes the solver search forever, so keep the default.
Worker processes run rubik.koc_worker, which does not import torch.
"""
from concurrent.futures import ProcessPoolExecutor

from .facelet import to_facelets
from .koc_worker import solve_facelets, solve_many

DEFAULT_MAX_DEPTH = 24

_solve_many = solve_many  # kept for older scripts


def solve_states(states, max_depth=DEFAULT_MAX_DEPTH, workers=1, chunk=64):
    """(N, 40) states -> list of solutions (lists of move indices or None)."""
    facelets = [to_facelets(row) for row in states.cpu()]
    if workers <= 1:
        return [solve_facelets(f, max_depth) for f in facelets]
    jobs = [(facelets[i:i + chunk], max_depth) for i in range(0, len(facelets), chunk)]
    with ProcessPoolExecutor(workers) as ex:
        return [sol for part in ex.map(solve_many, jobs) for sol in part]


def first_move_and_length(solutions):
    """Lists of (first move or -1, length or -1)."""
    first = [s[0] if s else -1 for s in solutions]
    length = [len(s) if s is not None else -1 for s in solutions]
    return first, length
