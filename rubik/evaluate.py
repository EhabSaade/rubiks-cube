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


def beam_rollout_batched(policy, states, width=16, max_moves=MAX_MOVES, cubes_per_batch=32):
    """Same search as beam_rollout, run for many cubes at once (faster on a GPU).

    Per cube: beams ranked by cumulative log-probability, states already in that
    cube's beam history forbidden, one copy per distinct state (its best score),
    the top `width` kept, stop at the first solved state or after max_moves.
    History and duplicates are matched on exact (cube, hash) pairs, so cubes
    never interact. Returns solved (N,) bool, length (N,) long, final (N, 40)."""
    n, dev = len(states), states.device
    solved_flag = is_solved(states)
    length = torch.where(solved_flag, 0, -1)
    best_final = states.clone()
    todo = torch.nonzero(~solved_flag).squeeze(1)
    for i in range(0, len(todo), cubes_per_batch):
        ids = todo[i:i + cubes_per_batch]
        k = len(ids)
        beam, bc = states[ids], torch.arange(k, device=dev)  # beam states and their cube (0..k-1)
        logp = torch.zeros(k, device=dev)
        hist_c, hist_h = bc.clone(), state_hash(beam)
        live = torch.ones(k, dtype=torch.bool, device=dev)
        for step in range(max_moves):
            lp = torch.log_softmax(policy(beam).float(), dim=1)  # (B, 18)
            nb = neighbors(beam).view(-1, STATE_DIM)
            cc = bc.repeat_interleave(N_MOVES)
            score = (logp.unsqueeze(1) + lp).view(-1)
            h = state_hash(nb)
            score = score.masked_fill(_pair_isin(cc, h, hist_c, hist_h), NEG)
            # group by cube, best score first within each cube
            order = torch.argsort(score, descending=True, stable=True)
            order = order[torch.argsort(cc[order], stable=True)]
            oc, oh = cc[order], h[order]
            first = _first_in_run(oc, oh)  # first (best) copy of each state within its cube
            valid = first & ~torch.isinf(score[order])
            counts = torch.bincount(oc[valid], minlength=k)
            base = torch.cumsum(counts, 0) - counts
            rank = torch.cumsum(valid.long(), 0) - 1 - base[oc]
            keep = order[valid & (rank < width)]
            # cubes with no move left fail; their final state is their current best beam state
            dead = live & (counts == 0)
            if dead.any():
                head = _first_per_cube(bc, k)
                for c in torch.nonzero(dead).squeeze(1).tolist():
                    best_final[ids[c]] = beam[head[c]]
                live &= ~dead
            beam, bc, logp = nb[keep], cc[keep], score[keep]
            hist_c, hist_h = torch.cat([hist_c, bc]), torch.cat([hist_h, h[keep]])
            hit = is_solved(beam)
            if hit.any():
                hit_cubes = torch.unique(bc[hit])
                first_hit = _first_per_cube(bc[hit], k)
                hit_states = beam[hit]
                for c in hit_cubes.tolist():
                    solved_flag[ids[c]] = True
                    length[ids[c]] = step + 1
                    best_final[ids[c]] = hit_states[first_hit[c]]
                live[hit_cubes] = False
            if not live.any():
                break
            stay = live[bc]
            beam, bc, logp = beam[stay], bc[stay], logp[stay]
        if live.any():
            head = _first_per_cube(bc, k)
            for c in torch.nonzero(live).squeeze(1).tolist():
                best_final[ids[c]] = beam[head[c]]
    return {"solved": solved_flag, "length": length, "final": best_final}


def _first_in_run(c, h):
    """For entries grouped by c (any order inside a group): True at the first
    entry of each distinct (c, h) pair, in the given order."""
    o = torch.argsort(h, stable=True)
    o = o[torch.argsort(c[o], stable=True)]
    cs, hs = c[o], h[o]
    new = torch.ones(len(o), dtype=torch.bool, device=c.device)
    new[1:] = (cs[1:] != cs[:-1]) | (hs[1:] != hs[:-1])
    out = torch.zeros(len(o), dtype=torch.bool, device=c.device)
    out[o[new]] = True  # stable sorts keep the earliest entry first in each run
    return out


def _pair_isin(qc, qh, rc, rh):
    """True where the (cube, hash) pair (qc, qh) occurs among the pairs (rc, rh)."""
    c, h = torch.cat([rc, qc]), torch.cat([rh, qh])
    is_ref = torch.zeros(len(c), dtype=torch.bool, device=c.device)
    is_ref[:len(rc)] = True
    o = torch.argsort(h, stable=True)
    o = o[torch.argsort(c[o], stable=True)]
    cs, hs = c[o], h[o]
    new = torch.ones(len(o), dtype=torch.bool, device=c.device)
    new[1:] = (cs[1:] != cs[:-1]) | (hs[1:] != hs[:-1])
    run = torch.cumsum(new.long(), 0) - 1
    has_ref = torch.zeros(int(run[-1]) + 1, dtype=torch.bool, device=c.device)
    has_ref[run[is_ref[o]]] = True
    out = torch.empty(len(c), dtype=torch.bool, device=c.device)
    out[o] = has_ref[run]
    return out[len(rc):]


def _first_per_cube(bc, k):
    """Index of the first entry of each cube 0..k-1 in bc (-1 if absent)."""
    pos = torch.arange(len(bc), device=bc.device)
    out = torch.full((k,), len(bc), dtype=torch.long, device=bc.device)
    out.scatter_reduce_(0, bc, pos, reduce="amin")
    return torch.where(out == len(bc), -1, out)


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


def shallow_search_policy(depth=2, heuristic=wrong_pieces, generator=None):
    """Score each move by the best heuristic value reachable within `depth`
    moves (the move itself plus depth-1 more); lower heuristic is better, so
    the score is its negative. Ties are broken randomly with `generator`."""
    g = generator if generator is not None else torch.Generator().manual_seed(0)

    def f(states):
        nb = neighbors(states)  # (k, 18, 40)
        k = len(states)
        level = nb.view(-1, STATE_DIM)
        best = heuristic(level).view(k, N_MOVES).float()
        for _ in range(depth - 1):
            level = neighbors(level).view(-1, STATE_DIM)
            h = heuristic(level).view(k, N_MOVES, -1).float().min(2).values
            best = torch.minimum(best, h)  # level grows 18x per depth; keep depth small
        noise = torch.rand(best.shape, generator=g).to(best.device)
        return -best + 1e-3 * noise  # seeded random tie-breaking
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
