import pytest
import torch

from rubik.cube import N_MOVES, apply_moves, random_states
from rubik.table import KNOWN_COUNTS, DistanceTable, state_hash

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def t4():
    return DistanceTable.build(4, DEV, verbose=False)


@pytest.fixture(scope="module")
def t3():
    return DistanceTable.build(3, DEV, verbose=False)


def test_counts(t4):
    counts = torch.bincount(t4.dist.long()).tolist()
    assert counts == KNOWN_COUNTS[:5]


def test_masks_are_exactly_the_optimal_moves(t4):
    s, d, m = t4.states, t4.dist.long(), t4.mask.long()
    assert (m[d == 0] == 0).all()
    assert (m[d > 0] != 0).all()
    for mv in range(N_MOVES):
        nd = t4.lookup(apply_moves(s, mv))
        bit = ((m >> mv) & 1).bool()
        assert (nd[bit] == d[bit] - 1).all()
        inner = d < 4  # neighbors of depth-4 states may leave the table
        assert (nd[~bit & inner] != d[~bit & inner] - 1).all()


def test_label_one_deeper_matches_bigger_table(t3, t4):
    s4 = t4.states[t4.dist == 4]
    d, m = t3.label(s4)
    assert (d == 4).all()
    assert torch.equal(m, t4.mask[t4.dist == 4])
    # inside the table, label equals the stored values
    d, m = t3.label(t3.states)
    assert torch.equal(d, t3.dist.long()) and torch.equal(m, t3.mask)


def test_unknown_states(t3):
    s = random_states(1000, torch.Generator().manual_seed(1), DEV)
    assert (t3.lookup(s) == -1).all()
    d, m = t3.label(s)
    assert (d == -1).all() and (m == 0).all()


def test_save_load(t3, tmp_path):
    p = tmp_path / "t.pt"
    t3.save(p)
    t = DistanceTable.load(p, DEV)
    assert torch.equal(t.keys, t3.keys) and torch.equal(t.mask, t3.mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_hash_same_on_cpu_and_gpu():
    s = random_states(1000)
    assert torch.equal(state_hash(s), state_hash(s.cuda()).cpu())
