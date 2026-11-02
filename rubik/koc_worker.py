"""Kociemba solving for worker processes, without importing torch.

On Windows every process that imports torch reserves several GB for the CUDA
libraries, so solver workers must stay torch-free.
"""
import kociemba

SOLVED_FACELETS = "".join(c * 9 for c in "URFDLB")
MOVE_NAMES = [f + p for f in "URFDLB" for p in ("", "2", "'")]


def solve_facelets(facelets, max_depth=24):
    """Facelet string -> list of move indices, or None if not found within max_depth."""
    if facelets == SOLVED_FACELETS:
        return []
    try:
        out = kociemba.solve(facelets, max_depth=max_depth)
    except ValueError:
        return None
    return [MOVE_NAMES.index(tok) for tok in out.split()]


def solve_many(args):
    facelets, max_depth = args
    return [solve_facelets(f, max_depth) for f in facelets]
