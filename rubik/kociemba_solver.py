"""Kociemba two-phase solver wrapper: solution, first move and length.

Lengths are upper bounds on true distance. max_depth is the two-phase search
limit: lower values give shorter solutions but take longer.
"""
from concurrent.futures import ProcessPoolExecutor

import kociemba

from .cube import parse_moves
from .facelet import SOLVED_FACELETS, to_facelets

DEFAULT_MAX_DEPTH = 24


def solve_facelets(facelets, max_depth=DEFAULT_MAX_DEPTH):
    """Facelet string -> list of move indices, or None if not found within max_depth."""
    if facelets == SOLVED_FACELETS:
        return []
    try:
        out = kociemba.solve(facelets, max_depth=max_depth)
    except ValueError:
        return None
    return parse_moves(out)


def _solve_many(args):
    facelets, max_depth = args
    return [solve_facelets(f, max_depth) for f in facelets]


def solve_states(states, max_depth=DEFAULT_MAX_DEPTH, workers=1, chunk=64):
    """(N, 40) states -> list of solutions (lists of move indices or None)."""
    facelets = [to_facelets(row) for row in states.cpu()]
    if workers <= 1:
        return [solve_facelets(f, max_depth) for f in facelets]
    jobs = [(facelets[i:i + chunk], max_depth) for i in range(0, len(facelets), chunk)]
    with ProcessPoolExecutor(workers) as ex:
        return [sol for part in ex.map(_solve_many, jobs) for sol in part]


def first_move_and_length(solutions):
    """Lists of (first move or -1, length or -1)."""
    first = [s[0] if s else -1 for s in solutions]
    length = [len(s) if s is not None else -1 for s in solutions]
    return first, length
