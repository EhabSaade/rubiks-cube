import torch

from rubik.exact_search import exact_label
from rubik.scramble import scramble
from rubik.table import DistanceTable

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_forward_search_matches_bigger_table():
    small = DistanceTable.build(1, DEV, verbose=False)  # labels exact to 2
    big = DistanceTable.build(4, DEV, verbose=False)    # labels exact to 5
    g = torch.Generator().manual_seed(3)
    for L in range(0, 7):
        states, _ = scramble(25, L, g, DEV)
        truth_d, truth_m = big.label(states)
        for s, t, tm in zip(states, truth_d, truth_m):
            d, m = exact_label(s, small, max_forward=3)  # exact to 5
            if t >= 0:
                assert d == int(t) and m == int(tm), (L, d, int(t), m, int(tm))
            else:
                assert (d, m) == (-1, 0)
