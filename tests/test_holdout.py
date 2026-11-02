import torch

from rubik.cube import random_states
from rubik.holdout import HoldoutFilter, make_holdout
from rubik.table import DistanceTable

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_holdout_share_determinism_and_filter():
    t = DistanceTable.build(4, DEV, verbose=False)
    a = make_holdout(t, depths=(3, 4), share=0.05, seed=1)
    b = make_holdout(t, depths=(3, 4), share=0.05, seed=1)
    assert torch.equal(a["states"], b["states"])
    assert int((a["dist"] == 3).sum()) == round(0.05 * 3240)
    assert int((a["dist"] == 4).sum()) == round(0.05 * 43239)
    f = HoldoutFilter(a["states"])
    inside = f.contains(t.states)
    assert int(inside.sum()) == len(a["states"])
    assert not f.contains(random_states(1000, device=DEV)).any()
