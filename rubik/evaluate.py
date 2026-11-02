"""Rollouts, progress measures and baseline policies.

A policy maps states (N, 40) uint8 to move scores (N, 18); higher is better.

Rollout rules (decision 22): at most 100 moves; a move whose result was
already visited in this rollout is forbidden and the next-best move is taken;
a cube with every move forbidden counts as failed.
"""
import torch

from .cube import N_MOVES, STATE_DIM, apply_moves, is_solved, neighbors, solved
from .encoding import encode
from .facelet import to_facelets
from .table import state_hash

MAX_MOVES = 100
NEG = float("-inf")


# ---------- progress measures ----------
def misplaced_cubies(states):
    """Number of the 20 pieces not in their home position. (N,) long."""
    s = states.long()
    home_c = torch.arange(8, device=s.device)
    home_e = torch.arange(12, device=s.device)
    return (s[:, 0:8] != home_c).sum(1) + (s[:, 16:28] != home_e).sum(1)


def misoriented_cubies(states):
    s = states.long()
    return (s[:, 8:16] != 0).sum(1) + (s[:, 28:40] != 0).sum(1)


def wrong_pieces(states):
    """Pieces that are misplaced or misoriented (0 only when solved)."""
    s = states.long()
    c = (s[:, 0:8] != torch.arange(8, device=s.device)) | (s[:, 8:16] != 0)
    e = (s[:, 16:28] != torch.arange(12, device=s.device)) | (s[:, 28:40] != 0)
    return c.sum(1) + e.sum(1)


def correct_stickers(states):
    """Stickers matching their face center, out of 48 non-center stickers."""
    out = []
    for row in states.cpu():
        f = to_facelets(row)
        out.append(sum(f[i] == f[9 * (i // 9) + 4] for i in range(54) if i % 9 != 4))
    return torch.tensor(out)


# ---------- rollouts ----------
def greedy_rollout(policy, states, max_moves=MAX_MOVES):
    """Returns dict with solved (N,) bool, length (N,) long (-1 if unsolved),
    final (N, 40) states and moves (N, max_moves) long (-1 padded)."""
    n, dev = len(states), states.device
    cur = states.clone()
    visited = torch.full((n, max_moves + 1), 0, dtype=torch.long, device=dev)
    visited[:, 0] = state_hash(cur)
    n_vis = 1
    done = is_solved(cur)
    failed = torch.zeros(n, dtype=torch.bool, device=dev)
    length = torch.where(done, 0, -1)
    moves = torch.full((n, max_moves), -1, dtype=torch.long, device=dev)
    for step in range(max_moves):
        active = ~done & ~failed
        if not active.any():
            break
        idx = torch.nonzero(active).squeeze(1)
        s = cur[idx]
        scores = policy(s).float()
        nb = neighbors(s)  # (k, 18, 40)
        nh = state_hash(nb.view(-1, STATE_DIM)).view(-1, N_MOVES)
        seen = (nh.unsqueeze(2) == visited[idx, :n_vis].unsqueeze(1)).any(2)
        scores = scores.masked_fill(seen, NEG)
        best = scores.argmax(1)
        stuck = torch.isinf(scores.max(1).values)
        nxt = nb[torch.arange(len(idx), device=dev), best]
        cur[idx] = torch.where(stuck.unsqueeze(1), s, nxt)
        moves[idx, step] = torch.where(stuck, -1, best)
        visited[idx, n_vis] = torch.where(stuck, 0, nh[torch.arange(len(idx), device=dev), best])
        n_vis += 1
        failed[idx[stuck]] = True
        now = is_solved(cur[idx]) & ~stuck
        done[idx[now]] = True
        length[idx[now]] = step + 1
    return {"solved": done, "length": length, "final": cur, "moves": moves}


def beam_rollout(policy, states, width=16, max_moves=MAX_MOVES):
    """Beam search on cumulative log-probability. States already in a cube's
    beam history are forbidden. Returns solved (N,) bool and length (N,)."""
    n, dev = len(states), states.device
    solved_flag = is_solved(states)
    length = torch.where(solved_flag, 0, -1)
    best_final = states.clone()
    for c in torch.nonzero(~solved_flag).squeeze(1).tolist():
        beam = states[c:c + 1]
        logp = torch.zeros(1, device=dev)
        hist = state_hash(beam)
        for step in range(max_moves):
            lp = torch.log_softmax(policy(beam).float(), dim=1)  # (b, 18)
            nb = neighbors(beam).view(-1, STATE_DIM)
            score = (logp.unsqueeze(1) + lp).view(-1)
            h = state_hash(nb)
            score = score.masked_fill(torch.isin(h, hist), NEG)
            # one copy per distinct state
            order = torch.argsort(score, descending=True)
            h_sorted = h[order]
            uniq, first_pos = _first_occurrence(h_sorted)
            keep = order[first_pos][:width]
            keep = keep[~torch.isinf(score[keep])]
            if len(keep) == 0:
                break
            beam, logp = nb[keep], score[keep]
            hist = torch.cat([hist, h[keep]])
            hit = is_solved(beam)
            if hit.any():
                solved_flag[c] = True
                length[c] = step + 1
                best_final[c] = beam[hit.nonzero()[0, 0]]
                break
        if not solved_flag[c]:
            best_final[c] = beam[0] if len(beam) else states[c]
    return {"solved": solved_flag, "length": length, "final": best_final}


def _first_occurrence(sorted_by_score_hashes):
    """Positions of the first occurrence of each hash in the given order."""
    h = sorted_by_score_hashes
    hs, perm = torch.sort(h, stable=True)
    first = torch.ones_like(hs, dtype=torch.bool)
    first[1:] = hs[1:] != hs[:-1]
    pos = perm[first].sort().values
    return hs[first], pos


# ---------- baseline and reference policies ----------
def random_policy(generator=None):
    def f(states):
        return torch.rand(len(states), N_MOVES, generator=generator).to(states.device)
    return f


def shallow_search_policy(depth=2, heuristic=wrong_pieces):
    """Score each move by the best heuristic value reachable within `depth`
    moves after it (lower heuristic is better, so the score is its negative)."""
    def f(states):
        nb = neighbors(states)  # (k, 18, 40)
        k = len(states)
        level = nb.view(-1, STATE_DIM)
        best = heuristic(level).view(k, N_MOVES).float()
        for _ in range(depth - 1):
            level = neighbors(level).view(-1, STATE_DIM)
            h = heuristic(level).view(k, N_MOVES, -1).float().min(2).values
            best = torch.minimum(best, h)  # level grows 18x per depth; keep depth small
        return -best + 1e-3 * torch.rand_like(best)  # random tie-breaking
    return f


def table_policy(table):
    """Oracle on the table range: score 1 for optimal moves, 0 otherwise."""
    bits = 1 << torch.arange(N_MOVES)
    def f(states):
        _, m = table.label(states)
        return ((m.long().unsqueeze(1) & bits.to(states.device)) != 0).float()
    return f


def model_policy(model, device="cuda"):
    @torch.no_grad()
    def f(states):
        return model(encode(states.to(device)))
    return f
