import torch

from rubik.batch_label import batch_label
from rubik.scramble import scramble
from rubik.table import DistanceTable

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_batch_label_matches_bigger_table():
    small = DistanceTable.build(1, DEV, verbose=False)  # labels exact to 2
    big = DistanceTable.build(4, DEV, verbose=False)    # labels exact to 5
    g = torch.Generator().manual_seed(5)
    for k in (1, 2, 3):
        for L in range(0, 7):
            s, _ = scramble(60, L, g, DEV)
            d, m = batch_label(s, small, k=k)  # exact to 2 + k
            td, tm = big.label(s)
            ok = (td >= 0) & (td <= 2 + k)
            assert torch.equal(d[ok], td[ok]) and torch.equal(m[ok], tm[ok])
            assert (d[~ok] == -1).all() and (m[~ok] == 0).all()
