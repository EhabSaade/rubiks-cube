import torch

from rubik.cube import N_MOVES, solved
from rubik.evaluate import (beam_rollout, beam_rollout_batched, correct_stickers, greedy_rollout,
                            misoriented_cubies, misplaced_cubies, random_policy, shallow_search_policy,
                            table_policy, wrong_pieces)
from rubik.scramble import scramble
from rubik.table import DistanceTable

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_oracle_policy_solves_optimally():
    t = DistanceTable.build(3, DEV, verbose=False)  # exact to 4
    states, _ = scramble(300, 4, torch.Generator().manual_seed(0), DEV)
    d, _ = t.label(states)
    pol = table_policy(t)
    g = greedy_rollout(pol, states)
    assert g["solved"].all() and torch.equal(g["length"], d)
    b = beam_rollout(pol, states[:30], width=4)
    assert b["solved"].all() and torch.equal(b["length"], d[:30])


def test_batched_beam_matches_beam():
    from rubik.encoding import encode
    from rubik.model import ResidualMLP
    torch.manual_seed(0)
    model = ResidualMLP(64, 1).to(DEV).eval()
    pol = lambda s: model(encode(s)).detach()  # untrained: continuous scores, no ties
    g = torch.Generator().manual_seed(6)
    states = torch.cat([scramble(12, d, g, DEV)[0] for d in (1, 2, 3, 8)])
    for width in (1, 8):
        a = beam_rollout(pol, states, width=width, max_moves=25)
        b = beam_rollout_batched(pol, states, width=width, max_moves=25, cubes_per_batch=5)
        assert torch.equal(a["solved"], b["solved"]) and torch.equal(a["length"], b["length"])
        assert torch.equal(a["final"], b["final"])
    t = DistanceTable.build(3, DEV, verbose=False)
    s4, _ = scramble(40, 4, torch.Generator().manual_seed(7), DEV)
    d, _ = t.label(s4)
    b = beam_rollout_batched(table_policy(t), s4, width=4, cubes_per_batch=16)
    assert b["solved"].all() and torch.equal(b["length"], d)


def test_rollout_cap_and_no_revisits():
    states, _ = scramble(50, 20, torch.Generator().manual_seed(1), DEV)
    g = greedy_rollout(random_policy(torch.Generator().manual_seed(2)), states, max_moves=30)
    assert ((g["length"] == -1) | (g["length"] <= 30)).all()
    # constant policy would loop U U U U; the no-revisit rule must stop that
    const = lambda s: torch.arange(N_MOVES, 0, -1, device=s.device).float().repeat(len(s), 1)
    g = greedy_rollout(const, states[:5], max_moves=10)
    for row in g["moves"]:
        mv = [m for m in row.tolist() if m >= 0]
        assert mv[:4] != [0, 0, 0, 0]


def test_heuristics_on_solved_and_scrambled():
    s = solved(1, DEV)
    assert misplaced_cubies(s).item() == 0 and misoriented_cubies(s).item() == 0
    assert wrong_pieces(s).item() == 0 and correct_stickers(s).item() == 48
    x, _ = scramble(20, 20, torch.Generator().manual_seed(3), DEV)
    assert (wrong_pieces(x) > 0).all() and (correct_stickers(x) < 48).all()


def test_shallow_search_solves_within_its_horizon_and_is_reproducible():
    # solved is visible within a 2-move horizon, so every 2-move scramble is solved
    states, _ = scramble(50, 2, torch.Generator().manual_seed(4), DEV)
    g = greedy_rollout(shallow_search_policy(2), states, max_moves=20)
    assert g["solved"].all()
    deeper, _ = scramble(50, 5, torch.Generator().manual_seed(5), DEV)
    a = greedy_rollout(shallow_search_policy(2), deeper, max_moves=30)
    b = greedy_rollout(shallow_search_policy(2), deeper, max_moves=30)
    assert torch.equal(a["moves"], b["moves"])
