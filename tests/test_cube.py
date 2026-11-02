import pytest
import torch

from rubik import cube
from rubik.cube import (INVERSE, N_MOVES, apply_moves, apply_sequence, invert_sequence,
                        is_solved, is_valid, neighbors, parse_moves, random_states, solved)
from rubik.encoding import ENCODING_DIM, encode
from rubik.facelet import SOLVED_FACELETS, from_facelets, to_facelets
from rubik.scramble import scramble

G = torch.Generator().manual_seed(0)

SUPERFLIP_SEQ = "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"
SUPERFLIP_FACELETS = "UBULURUFURURFRBRDRFUFLFRFDFDFDLDRDBDLULBLFLDLBUBRBLBDB"


def test_move_then_inverse_is_identity():
    s = random_states(500, G)
    for m in range(N_MOVES):
        back = apply_moves(apply_moves(s, m), int(INVERSE[m]))
        assert torch.equal(back, s)


def test_move_orders():
    s = solved()
    for m in range(N_MOVES):
        order = 2 if m % 3 == 1 else 4
        t = s
        for k in range(1, order + 1):
            t = apply_moves(t, m)
            assert bool(is_solved(t)) == (k == order), (cube.MOVE_NAMES[m], k)


def test_sexy_move_has_order_6():
    seq = parse_moves("R U R' U'")
    t = solved()
    for k in range(1, 7):
        t = apply_sequence(t, seq)
        assert bool(is_solved(t)) == (k == 6)


def test_single_R_facelets():
    t = apply_sequence(solved(), parse_moves("R"))
    assert to_facelets(t[0]) == "UUFUUFUUFRRRRRRRRRFFDFFDFFDDDBDDBDDBLLLLLLLLLUBBUBBUBB"


def test_superflip():
    t = apply_sequence(solved(), parse_moves(SUPERFLIP_SEQ))
    row = t[0].long()
    assert torch.equal(row[0:8], torch.arange(8))
    assert (row[8:16] == 0).all()
    assert torch.equal(row[16:28], torch.arange(12))
    assert (row[28:40] == 1).all()
    assert to_facelets(t[0]) == SUPERFLIP_FACELETS


def test_facelet_roundtrip():
    assert to_facelets(solved()[0]) == SOLVED_FACELETS
    s = random_states(200, G)
    for row in s:
        assert torch.equal(from_facelets(to_facelets(row))[0], row)


def test_random_states_valid_and_scrambles_valid():
    assert is_valid(random_states(2000, G)).all()
    states, _ = scramble(2000, 25, G)
    assert is_valid(states).all()


def test_scramble_undo():
    states, moves = scramble(50, 20, G)
    for st, mv in zip(states, moves):
        back = apply_sequence(st.unsqueeze(0), invert_sequence(mv.tolist()))
        assert is_solved(back).all()


def test_neighbors_match_apply_moves():
    s = random_states(10, G)
    nb = neighbors(s)
    for m in range(N_MOVES):
        assert torch.equal(nb[:, m], apply_moves(s, m))


def test_bfs_counts_to_depth_3():
    # known half-turn metric counts: 1, 18, 243, 3240
    seen = solved()
    frontier = solved()
    for expected in (18, 243, 3240):
        nb = neighbors(frontier).view(-1, 40)
        nb = torch.unique(nb, dim=0)
        both = torch.cat([seen, nb])
        uniq, inv, counts = torch.unique(both, dim=0, return_inverse=True, return_counts=True)
        new_mask = counts[inv[len(seen):]] == 1
        frontier = nb[new_mask]
        assert len(frontier) == expected
        seen = torch.cat([seen, frontier])


def test_encoding():
    s = random_states(100, G)
    e = encode(s)
    assert e.shape == (100, ENCODING_DIM)
    assert (e.sum(1) == 20).all()
    # each piece block of 24 has exactly one hot value
    assert (e.view(100, 20, 24).sum(2) == 1).all()
    # distinct states give distinct encodings
    s2 = apply_moves(s, 0)
    assert not (encode(s2) == e).all(1).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_gpu_matches_cpu():
    s = random_states(1000, G)
    moves = torch.randint(0, N_MOVES, (1000,), generator=G)
    cpu = apply_moves(s, moves)
    gpu = apply_moves(s.cuda(), moves.cuda()).cpu()
    assert torch.equal(cpu, gpu)
    assert torch.equal(encode(s), encode(s.cuda()).cpu())


def test_kociemba_solution_replays():
    kociemba = pytest.importorskip("kociemba")
    s = random_states(20, G)
    for row in s:
        sol = kociemba.solve(to_facelets(row))
        out = apply_sequence(row.unsqueeze(0), parse_moves(sol))
        assert is_solved(out).all()
