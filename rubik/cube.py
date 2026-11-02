"""Batched 3x3 cube simulator on the cubie level.

Conventions follow Kociemba's two-phase solver so states convert directly to
its facelet strings.

State layout: uint8 tensor of shape (N, 40)
    [0:8]   corner permutation  cp[i] = corner piece sitting at position i
    [8:16]  corner orientation  co[i] in {0,1,2}
    [16:28] edge permutation    ep[i] = edge piece sitting at position i
    [28:40] edge orientation    eo[i] in {0,1}

Moves are indexed face*3 + power, faces U R F D L B, powers (cw, half, ccw):
    0:U 1:U2 2:U' 3:R 4:R2 5:R' ... 15:B 16:B2 17:B'
Applying move m to state s is s*m (Kociemba's "is replaced by" product).
"""
import torch

URF, UFL, ULB, UBR, DFR, DLF, DBL, DRB = range(8)
UR, UF, UL, UB, DR, DF, DL, DB, FR, FL, BL, BR = range(12)

FACES = "URFDLB"
POWERS = ["", "2", "'"]
MOVE_NAMES = [f + p for f in FACES for p in POWERS]
N_MOVES = 18
STATE_DIM = 40

# Kociemba's six basic clockwise face turns: (cp, co, ep, eo)
_BASIC = {
    "U": ([UBR, URF, UFL, ULB, DFR, DLF, DBL, DRB], [0] * 8,
          [UB, UR, UF, UL, DR, DF, DL, DB, FR, FL, BL, BR], [0] * 12),
    "R": ([DFR, UFL, ULB, URF, DRB, DLF, DBL, UBR], [2, 0, 0, 1, 1, 0, 0, 2],
          [FR, UF, UL, UB, BR, DF, DL, DB, DR, FL, BL, UR], [0] * 12),
    "F": ([UFL, DLF, ULB, UBR, URF, DFR, DBL, DRB], [1, 2, 0, 0, 2, 1, 0, 0],
          [UR, FL, UL, UB, DR, FR, DL, DB, UF, DF, BL, BR], [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]),
    "D": ([URF, UFL, ULB, UBR, DLF, DBL, DRB, DFR], [0] * 8,
          [UR, UF, UL, UB, DF, DL, DB, DR, FR, FL, BL, BR], [0] * 12),
    "L": ([URF, ULB, DBL, UBR, DFR, UFL, DLF, DRB], [0, 1, 2, 0, 0, 2, 1, 0],
          [UR, UF, BL, UB, DR, DF, FL, DB, FR, UL, DL, BR], [0] * 12),
    "B": ([URF, UFL, UBR, DRB, DFR, DLF, ULB, DBL], [0, 0, 1, 2, 0, 0, 2, 1],
          [UR, UF, UL, BR, DR, DF, DL, BL, FR, FL, UB, DB], [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1]),
}


def _multiply(a, b):
    """Single-cube product a*b on (cp, co, ep, eo) lists."""
    acp, aco, aep, aeo = a
    bcp, bco, bep, beo = b
    cp = [acp[bcp[i]] for i in range(8)]
    co = [(aco[bcp[i]] + bco[i]) % 3 for i in range(8)]
    ep = [aep[bep[i]] for i in range(12)]
    eo = [(aeo[bep[i]] + beo[i]) % 2 for i in range(12)]
    return cp, co, ep, eo


def _build_tables():
    cps, cos, eps, eos = [], [], [], []
    for f in FACES:
        m = _BASIC[f]
        p = m
        for _ in range(3):
            cps.append(p[0]); cos.append(p[1]); eps.append(p[2]); eos.append(p[3])
            p = _multiply(p, m)
    return (torch.tensor(cps), torch.tensor(cos), torch.tensor(eps), torch.tensor(eos))


_CPU_TABLES = _build_tables()
_TABLE_CACHE = {}

# INVERSE[m] undoes move m; FACE_OF[m] is the face index of move m
INVERSE = torch.tensor([f * 3 + (2 - p) for f in range(6) for p in range(3)])
FACE_OF = torch.arange(N_MOVES) // 3


def _tables(device):
    key = str(device)
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = tuple(t.to(device) for t in _CPU_TABLES)
    return _TABLE_CACHE[key]


def solved(n=1, device="cpu"):
    """n copies of the solved cube."""
    row = torch.cat([torch.arange(8), torch.zeros(8, dtype=torch.long),
                     torch.arange(12), torch.zeros(12, dtype=torch.long)])
    return row.to(torch.uint8).to(device).repeat(n, 1)


def is_solved(state):
    """Bool tensor (N,), True where the cube is solved."""
    return (state == solved(1, state.device)).all(dim=1)


def apply_moves(state, moves):
    """Apply one move per cube. state (N,40) uint8, moves (N,) long or int."""
    if isinstance(moves, int):
        moves = torch.full((state.shape[0],), moves, dtype=torch.long, device=state.device)
    tcp, tco, tep, teo = _tables(state.device)
    s = state.long()
    cp, co, ep, eo = s[:, 0:8], s[:, 8:16], s[:, 16:28], s[:, 28:40]
    mcp, mep = tcp[moves], tep[moves]
    out = torch.cat([
        cp.gather(1, mcp),
        (co.gather(1, mcp) + tco[moves]) % 3,
        ep.gather(1, mep),
        (eo.gather(1, mep) + teo[moves]) % 2,
    ], dim=1)
    return out.to(torch.uint8)


def apply_sequence(state, seq):
    """Apply the same move sequence (list of ints) to every cube."""
    for m in seq:
        state = apply_moves(state, int(m))
    return state


def apply_move_matrix(state, moves):
    """Apply per-cube sequences. moves (N, L) long."""
    for j in range(moves.shape[1]):
        state = apply_moves(state, moves[:, j])
    return state


def neighbors(state):
    """All 18 successors of each cube: (N, 18, 40)."""
    n = state.shape[0]
    rep = state.repeat_interleave(N_MOVES, dim=0)
    moves = torch.arange(N_MOVES, device=state.device).repeat(n)
    return apply_moves(rep, moves).view(n, N_MOVES, STATE_DIM)


def parse_moves(text):
    """'R U2 F'' -> [3, 1, 8]."""
    return [MOVE_NAMES.index(tok) for tok in text.split()]


def format_moves(seq):
    return " ".join(MOVE_NAMES[int(m)] for m in seq)


def invert_sequence(seq):
    return [int(INVERSE[int(m)]) for m in reversed(seq)]


def _parity(perm):
    """Permutation parity per row (0 even, 1 odd). perm (N, k) long."""
    k = perm.shape[1]
    i, j = torch.triu_indices(k, k, offset=1, device=perm.device)
    return ((perm[:, i] > perm[:, j]).sum(dim=1) % 2)


def is_valid(state):
    """Checks permutation, orientation sums and parity."""
    s = state.long()
    cp, co, ep, eo = s[:, 0:8], s[:, 8:16], s[:, 16:28], s[:, 28:40]
    perm_ok = ((cp.sort(dim=1).values == torch.arange(8, device=s.device)).all(1)
               & (ep.sort(dim=1).values == torch.arange(12, device=s.device)).all(1))
    return (perm_ok & (co.sum(1) % 3 == 0) & (eo.sum(1) % 2 == 0)
            & (_parity(cp) == _parity(ep)))


def random_states(n, generator=None, device="cpu"):
    """Uniformly random reachable cube states."""
    g = generator
    cp = torch.argsort(torch.rand(n, 8, generator=g), dim=1)
    ep = torch.argsort(torch.rand(n, 12, generator=g), dim=1)
    # fix parity by swapping the last two edges where it disagrees
    bad = _parity(cp) != _parity(ep)
    ep[bad, 10], ep[bad, 11] = ep[bad, 11].clone(), ep[bad, 10].clone()
    co = torch.randint(0, 3, (n, 8), generator=g)
    co[:, 7] = (-co[:, :7].sum(1)) % 3
    eo = torch.randint(0, 2, (n, 12), generator=g)
    eo[:, 11] = eo[:, :11].sum(1) % 2
    return torch.cat([cp, co, ep, eo], dim=1).to(torch.uint8).to(device)
