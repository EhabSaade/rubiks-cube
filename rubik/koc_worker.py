"""Kociemba solving for worker processes, without importing torch.

On Windows every process that imports torch reserves several GB for the CUDA
libraries. multiprocessing workers re-import the main script (which imports
torch), so workers are started as plain subprocesses running this module:

    python -m rubik.koc_worker <max_depth>   # facelet strings on stdin, one per line

Each output line is the solution as space-separated move indices, "" for an
already solved cube, or "None" if no solution was found within max_depth.
"""
import sys

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


def encode_solution(sol):
    return "None" if sol is None else " ".join(map(str, sol))


def decode_solution(line):
    line = line.strip()
    if line == "None":
        return None
    return [int(x) for x in line.split()] if line else []


def main():
    max_depth = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    out = sys.stdout
    for line in sys.stdin:
        out.write(encode_solution(solve_facelets(line.strip(), max_depth)) + "\n")
    out.flush()


if __name__ == "__main__":
    main()
