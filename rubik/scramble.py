"""Scramble generator.

Random move sequences without obvious redundancy: no two consecutive turns of
the same face, and turns of opposite faces (which commute) only in one fixed
order, e.g. "U D" allowed but "D U" not. Scramble length is an upper bound on
distance, not the true distance.
"""
import torch

from .cube import FACE_OF, N_MOVES, apply_move_matrix, solved


def _allowed(prev_face, face):
    if prev_face is None:
        return torch.ones_like(face, dtype=torch.bool)
    same = face == prev_face
    # opposite faces: U/D, R/L, F/B are 3 apart; require ascending face order
    opposite_wrong_order = (face % 3 == prev_face % 3) & (face < prev_face)
    return ~(same | opposite_wrong_order)


def random_move_sequences(n, length, generator=None):
    """(n, length) long tensor of non-redundant random moves."""
    moves = torch.empty(n, length, dtype=torch.long)
    prev = None
    for j in range(length):
        col = torch.randint(0, N_MOVES, (n,), generator=generator)
        ok = _allowed(prev, FACE_OF[col])
        while not ok.all():
            k = int((~ok).sum())
            col[~ok] = torch.randint(0, N_MOVES, (k,), generator=generator)
            ok = _allowed(prev, FACE_OF[col])
        moves[:, j] = col
        prev = FACE_OF[col]
    return moves


def scramble(n, length, generator=None, device="cpu"):
    """Returns (states (n,40), moves (n,length)) scrambled from solved."""
    moves = random_move_sequences(n, length, generator)
    states = apply_move_matrix(solved(n, device), moves.to(device))
    return states, moves
