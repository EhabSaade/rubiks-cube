"""Conversion between cubie states and Kociemba 54-character facelet strings.

Facelet order is U1..U9 R1..R9 F1..F9 D1..D9 L1..L9 B1..B9.
"""
import torch

from .cube import STATE_DIM

U1, R1, F1, D1, L1, B1 = 0, 9, 18, 27, 36, 45
SOLVED_FACELETS = "".join(c * 9 for c in "URFDLB")


def _f(face_start, k):
    return face_start + k - 1


CORNER_FACELET = [
    [_f(U1, 9), _f(R1, 1), _f(F1, 3)], [_f(U1, 7), _f(F1, 1), _f(L1, 3)],
    [_f(U1, 1), _f(L1, 1), _f(B1, 3)], [_f(U1, 3), _f(B1, 1), _f(R1, 3)],
    [_f(D1, 3), _f(F1, 9), _f(R1, 7)], [_f(D1, 1), _f(L1, 9), _f(F1, 7)],
    [_f(D1, 7), _f(B1, 9), _f(L1, 7)], [_f(D1, 9), _f(R1, 9), _f(B1, 7)],
]
EDGE_FACELET = [
    [_f(U1, 6), _f(R1, 2)], [_f(U1, 8), _f(F1, 2)], [_f(U1, 4), _f(L1, 2)], [_f(U1, 2), _f(B1, 2)],
    [_f(D1, 6), _f(R1, 8)], [_f(D1, 2), _f(F1, 8)], [_f(D1, 4), _f(L1, 8)], [_f(D1, 8), _f(B1, 8)],
    [_f(F1, 6), _f(R1, 4)], [_f(F1, 4), _f(L1, 6)], [_f(B1, 6), _f(L1, 4)], [_f(B1, 4), _f(R1, 6)],
]
CORNER_COLOR = ["URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB"]
EDGE_COLOR = ["UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR"]


def to_facelets(state_row):
    """One state (40,) -> facelet string."""
    s = [int(x) for x in state_row]
    cp, co, ep, eo = s[0:8], s[8:16], s[16:28], s[28:40]
    f = ["?"] * 54
    for i in range(8):
        for n in range(3):
            f[CORNER_FACELET[i][(n + co[i]) % 3]] = CORNER_COLOR[cp[i]][n]
    for i in range(12):
        for n in range(2):
            f[EDGE_FACELET[i][(n + eo[i]) % 2]] = EDGE_COLOR[ep[i]][n]
    for k, c in enumerate("URFDLB"):
        f[9 * k + 4] = c
    return "".join(f)


def from_facelets(text):
    """Facelet string -> state (1, 40) uint8. Raises ValueError if malformed."""
    if len(text) != 54:
        raise ValueError("facelet string must have 54 characters")
    cp, co, ep, eo = [0] * 8, [0] * 8, [0] * 12, [0] * 12
    for i in range(8):
        ori = next((o for o in range(3) if text[CORNER_FACELET[i][o]] in "UD"), None)
        if ori is None:
            raise ValueError(f"corner {i} has no U/D sticker")
        c1 = text[CORNER_FACELET[i][(ori + 1) % 3]]
        c2 = text[CORNER_FACELET[i][(ori + 2) % 3]]
        j = next((j for j in range(8) if CORNER_COLOR[j][1] == c1 and CORNER_COLOR[j][2] == c2), None)
        if j is None:
            raise ValueError(f"corner {i} has invalid colors")
        cp[i], co[i] = j, ori
    for i in range(12):
        a, b = text[EDGE_FACELET[i][0]], text[EDGE_FACELET[i][1]]
        for j in range(12):
            if EDGE_COLOR[j] == a + b:
                ep[i], eo[i] = j, 0
                break
            if EDGE_COLOR[j] == b + a:
                ep[i], eo[i] = j, 1
                break
        else:
            raise ValueError(f"edge {i} has invalid colors")
    return torch.tensor([cp + co + ep + eo], dtype=torch.uint8).view(1, STATE_DIM)
