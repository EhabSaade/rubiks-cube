import torch

from rubik.exact_search import exact_distance
from rubik.scramble import scramble
from rubik.table import DistanceTable

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_forward_search_matches_bigger_table():
    small = DistanceTable.build(2, DEV, verbose=False)  # labels exact to 3
    big = DistanceTable.build(4, DEV, verbose=False)    # labels exact to 5
    g = torch.Generator().manual_seed(3)
    for L in range(0, 7):
        states, _ = scramble(15, L, g, DEV)
        truth, _ = big.label(states)
        for s, t in zip(states, truth):
            got = exact_distance(s, small, max_forward=2)  # exact to 5
            if t >= 0:
                assert got == int(t)
            else:
                assert got == -1
