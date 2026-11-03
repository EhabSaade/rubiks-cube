import torch

from rubik.encoding import encode
from rubik.model import ResidualMLP
from rubik.pool import ARMS, CurriculumSampler, Pool, soft_targets
from rubik.table import DistanceTable

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def toy_pool(n=1000):
    g = torch.Generator().manual_seed(0)
    true_key = torch.randint(1, 13, (n,), generator=g)
    koc_key = true_key + torch.randint(0, 6, (n,), generator=g)
    states = torch.zeros(n, 40, dtype=torch.uint8)
    return Pool(states, torch.ones(n, dtype=torch.int32), true_key, koc_key, torch.ones(n, dtype=torch.bool))


def test_from_parts_targets_and_keys():
    part = {"states": torch.zeros(3, 40, dtype=torch.uint8),
            "exact_dist": torch.tensor([8, -1, 9]), "exact_mask": torch.tensor([0b110, 0, 0b1]),
            "koc_move": torch.tensor([4, 5, 0]), "koc_len": torch.tensor([10, 14, 9])}
    p = Pool.from_parts([part])
    assert p.mask.tolist() == [0b110, 1 << 5, 0b1]  # exact where known, else Kociemba move
    assert p.true_key.tolist() == [8, 14, 9]          # Kociemba length where distance unknown
    assert p.koc_key.tolist() == [10, 14, 9]
    assert p.exact.tolist() == [True, False, True]


def test_equal_share_stages_same_for_all_arms():
    pool = toy_pool()
    samplers = {a: CurriculumSampler(pool, a, n_stages=5, total_steps=100, seed=1) for a in ARMS}
    for a, s in samplers.items():
        assert s.steps_per_stage == 20 and s.stage(0) == 0 and s.stage(99) == 4
        lens = [s.prefix_len(st) for st in (0, 20, 40, 60, 80)]
        assert lens == ([1000] * 5 if a == "mixed" else [200, 400, 600, 800, 1000])


def test_stage_prefix_is_ordered_by_the_arms_key():
    pool = toy_pool()
    for arm, key in (("true_distance", pool.true_key), ("kociemba", pool.koc_key)):
        s = CurriculumSampler(pool, arm, n_stages=5, total_steps=100, seed=1)
        first = s.order[:200]
        rest = s.order[200:]
        assert key[first].max() <= key[rest].min()
        idx = s.sample(0, 5000)
        assert torch.isin(idx, first).all()


def test_soft_targets():
    t = soft_targets(torch.tensor([0b101, 1 << 17]))
    assert torch.allclose(t.sum(1), torch.ones(2))
    assert t[0, 0] == 0.5 and t[0, 2] == 0.5 and t[1, 17] == 1.0


def test_model_overfits_tiny_batch():
    torch.manual_seed(0)
    table = DistanceTable.build(3, DEV, verbose=False)
    sel = table.dist > 0
    s, m = table.states[sel][:256], table.mask[sel][:256]
    model = ResidualMLP(width=256, blocks=2).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = encode(s), soft_targets(m)
    for _ in range(300):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    hit = ((m.long() >> model(x).argmax(1)) & 1).float().mean()
    assert hit > 0.98
