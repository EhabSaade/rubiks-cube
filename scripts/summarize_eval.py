"""Summarize results/final_eval/*.json by arm (mean and range over seeds), per
test set, next to the baselines; for the primary (step 100,000) and the
secondary (best-validation) checkpoints.

    python scripts/summarize_eval.py      # writes results/final_eval/summary.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "final_eval"
ARMS = ["true_distance", "kociemba", "mixed"]
SETS = ["d1_4_all", "holdout_d5", "holdout_d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "random", "d20_known"]
METRICS = [("optimal_move_rate", None, "Optimal-move rate"),
           ("solve_rate", "greedy", "Greedy solve rate"),
           ("solve_rate", "beam", "Beam (4096) solve rate"),
           ("mean_length_solved", "beam", "Beam mean solution length (solved)"),
           ("mean_excess_over_exact", "beam", "Beam mean excess over exact distance (solved)"),
           ("mean_excess_over_kociemba", "beam", "Beam mean excess over Kociemba length (solved)"),
           ("mean_length_solved", "greedy", "Greedy mean solution length (solved)"),
           ("unsolved_wrong_pieces", "greedy", "Greedy unsolved: wrong pieces (of 20)"),
           ("seconds_per_cube", "beam", "Beam seconds per cube"),
           ("seconds_per_cube", "greedy", "Greedy seconds per cube")]
PCT = {"optimal_move_rate", "solve_rate"}


def load(kind):
    """{arm: {seed: result}} for kind 'final' or 'bestval'."""
    out = {a: {} for a in ARMS}
    for arm in ARMS:
        for s in range(3):
            run = f"final_{arm}_s{s}"
            p = EVAL / (f"{run}.json" if kind == "final" else f"{run}_bestval.json")
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            if "same_as_final" in r:
                f = EVAL / r["same_as_final"]
                if not f.exists():
                    continue
                r = dict(json.loads(f.read_text()), checkpoint="best_val (= final)")
            out[arm][s] = r
    return out


def value(r, name, metric, part):
    e = r["sets"].get(name)
    if e is None:
        return None
    return e[metric] if part is None else e[part][metric]


def fmt(vals, pct):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    m = sum(vals) / len(vals)
    if pct:
        s = f"{m:.1%}"
        return s if len(vals) == 1 or max(vals) - min(vals) < 0.0005 else f"{s} ({min(vals):.1%}-{max(vals):.1%})"
    s = f"{m:.2f}"
    return s if len(vals) == 1 or max(vals) - min(vals) < 0.005 else f"{s} ({min(vals):.2f}-{max(vals):.2f})"


def baseline_rows():
    rows = json.loads((ROOT / "results" / "baselines.json").read_text())
    return {(r["set"], r["policy"]): r for r in rows}


def table(res, metric, part, title, base):
    pct = metric in PCT
    head = ["Set"] + [f"{a} (n={len(res[a])})" for a in ARMS]
    bcols = []
    if part in ("greedy", None) and metric in ("optimal_move_rate", "solve_rate", "mean_length_solved"):
        bcols = {"optimal_move_rate": ["kociemba", "shallow_search_d2", "random"],
                 "solve_rate": ["kociemba", "shallow_search_d2"], "mean_length_solved": ["kociemba"]}[metric]
        head += [f"baseline: {b}" for b in bcols]
    lines = [f"### {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name in SETS:
        cells = [name] + [fmt([value(r, name, metric, part) for r in res[a].values()], pct) for a in ARMS]
        for b in bcols:
            v = base.get((name, b), {}).get(metric)
            cells.append("-" if v is None else (f"{v:.1%}" if pct else f"{v:.2f}"))
        lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def main():
    base = baseline_rows()
    md = ["# Final evaluation on the frozen test sets", "",
              "Mean over seeds, with the range in brackets when the seeds differ. "
              "Rollouts use the baselines' fixed subsample of at most 1,000 cubes per set; "
              "the optimal-move rate uses every cube with known optimal moves.", ""]
    for kind, title in (("final", "Primary: step-100,000 checkpoints"),
                        ("bestval", "Secondary: best-validation checkpoints")):
        res = load(kind)
        steps = {a: {s: r["step"] for s, r in res[a].items()} for a in ARMS}
        md += [f"## {title}", "", f"Checkpoint steps by arm and seed: {json.dumps(steps)}", ""]
        for metric, part, t in METRICS:
            md += table(res, metric, part, t, base)
    (EVAL / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
