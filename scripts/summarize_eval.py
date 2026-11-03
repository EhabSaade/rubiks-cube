"""Summarize every evaluation into results/summary.md and results/eval_tidy.csv.

    python scripts/summarize_eval.py

Groups: the main runs (final_<arm>_s<seed>, 20 stages; results/final_eval/),
the 2-stage curricula (stages2_<arm>_s<seed>; results/stages2_eval/) and the
beam-width sweep (results/beam_sweep/). Per arm: mean over seeds, with the
range over seeds in brackets. Key comparisons: difference of means with a 95%
Welch t-interval over seeds (each seed's result is one observation per arm).
Solve times come from the clean re-timing (results/final_eval/timing/); the
seconds per cube inside the evaluation files were measured with GPUs shared and
are not used. The tidy CSV has one row per run, checkpoint, set, method, width
and metric.
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SETS = ["d1_4_all", "holdout_d5", "holdout_d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "random", "d20_known"]
DEEP = ["d8", "d9", "d10", "d11", "d12", "d13", "random", "d20_known"]
ARMS = ["true_distance", "kociemba", "mixed"]
PCT = {"optimal_move_rate", "solve_rate"}
ORDER = ["true_distance (20 stages)", "kociemba (20 stages)", "true_distance (2 stages)",
         "kociemba (2 stages)", "mixed"]
ROLL_METRICS = ["solve_rate", "mean_length_solved", "mean_excess_over_exact", "mean_excess_over_kociemba",
                "unsolved_misplaced", "unsolved_misoriented", "unsolved_wrong_pieces", "unsolved_correct_stickers"]


# ---------- loading ----------
def load_dir(d, prefix, kind):
    """{arm: {seed: result}}; kind 'final' or 'bestval' (pointers resolved)."""
    out = {}
    if not d.exists():
        return out
    pat = re.compile(rf"^{prefix}_(\w+?)_s(\d+)(_bestval)?\.json$")
    for p in sorted(d.glob(f"{prefix}_*.json")):
        m = pat.match(p.name)
        if not m or (m.group(3) is not None) != (kind == "bestval"):
            continue
        r = json.loads(p.read_text())
        if "same_as_final" in r:
            f = d / r["same_as_final"]
            if not f.exists():
                continue
            r = dict(json.loads(f.read_text()), checkpoint="best_val (= final)")
        out.setdefault(m.group(1), {})[int(m.group(2))] = r
    return out


def _label(prefix, arm):
    if arm == "mixed":
        return "mixed"
    return f"{arm} ({'20' if prefix == 'final' else '2'} stages)"


def groups(kind):
    """Arm label -> {seed: evaluation result}."""
    g = {}
    for d, prefix in ((RES / "final_eval", "final"), (RES / "stages2_eval", "stages2")):
        for arm, runs in load_dir(d, prefix, kind).items():
            g[_label(prefix, arm)] = runs
    return {k: g[k] for k in ORDER if k in g}


def sweep_groups(kind):
    g = {}
    for prefix in ("final", "stages2"):
        for arm, runs in load_dir(RES / "beam_sweep", prefix, kind).items():
            g[_label(prefix, arm)] = runs
    return {k: g[k] for k in ORDER if k in g}


# ---------- formatting ----------
def fmt(vals, pct, digits=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    f = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.{digits}f}")
    s = f(float(np.mean(vals)))
    if len(vals) > 1 and f(min(vals)) != f(max(vals)):
        s += f" ({f(min(vals))}-{f(max(vals))})"
    return s


def md_table(head, rows):
    return (["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            + ["| " + " | ".join(r) + " |" for r in rows] + [""])


def val(r, name, metric, part):
    e = r["sets"].get(name)
    if e is None:
        return None
    return e.get(metric) if part is None else e[part].get(metric)


# two-sided 95% t quantiles by degrees of freedom (1 to 30); 1.96 beyond
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
        2.045, 2.042]


def welch_diff(a, b):
    """Mean(a) - mean(b) with a 95% Welch t-interval (df rounded down, so slightly conservative)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    d, se = a.mean() - b.mean(), np.sqrt(va + vb)
    if se == 0:
        return d, d, d, float("inf")
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = T975[min(int(df), 30) - 1] if df >= 1 else T975[0]
    t = 1.96 if df > 30 else t
    return d, d - t * se, d + t * se, abs(d) / se


# ---------- sections ----------
METRICS = [("optimal_move_rate", None, "Optimal-move rate"),
           ("solve_rate", "greedy", "Greedy solve rate"),
           ("solve_rate", "beam", "Beam (4096) solve rate"),
           ("mean_length_solved", "beam", "Beam (4096) mean solution length (solved)"),
           ("mean_excess_over_exact", "beam", "Beam (4096) mean excess over exact distance (solved)"),
           ("mean_excess_over_kociemba", "beam", "Beam (4096) mean excess over Kociemba length (solved)"),
           ("mean_length_solved", "greedy", "Greedy mean solution length (solved)"),
           ("unsolved_wrong_pieces", "greedy", "Greedy, unsolved cubes: wrong pieces (of 20)"),
           ("unsolved_correct_stickers", "greedy", "Greedy, unsolved cubes: correct stickers (of 48)")]


def baselines():
    rows = json.loads((RES / "baselines.json").read_text())
    return {(r["set"], r["policy"]): r for r in rows}


def eval_section(g, base):
    md = []
    for metric, part, title in METRICS:
        bcols = {"optimal_move_rate": ["kociemba", "shallow_search_d2", "random"],
                 "solve_rate": ["shallow_search_d2"] if part == "greedy" else [],
                 "mean_length_solved": ["kociemba"]}.get(metric, [])
        head = ["Set"] + [f"{a} (n={len(g[a])})" for a in g] + [f"baseline: {b}" for b in bcols]
        rows = []
        for name in SETS:
            row = [name] + [fmt([val(r, name, metric, part) for r in g[a].values()], metric in PCT) for a in g]
            for b in bcols:
                v = base.get((name, b), {}).get(metric)
                row.append("-" if v is None else (f"{v:.1%}" if metric in PCT else f"{v:.2f}"))
            rows.append(row)
        md += [f"### {title}", ""] + md_table(head, rows)
    return md


COMPARE = [("greedy solve rate, d10", "solve_rate", "greedy", "d10", True),
           ("greedy solve rate, d13", "solve_rate", "greedy", "d13", True),
           ("greedy solve rate, random", "solve_rate", "greedy", "random", True),
           ("optimal-move rate, d10", "optimal_move_rate", None, "d10", True),
           ("optimal-move rate, d13", "optimal_move_rate", None, "d13", True),
           ("beam excess over exact, d13", "mean_excess_over_exact", "beam", "d13", False),
           ("beam mean length, random", "mean_length_solved", "beam", "random", False),
           ("beam mean length, d20_known", "mean_length_solved", "beam", "d20_known", False)]
PAIRS = [("mixed", "true_distance (20 stages)"), ("mixed", "kociemba (20 stages)"),
         ("kociemba (20 stages)", "true_distance (20 stages)"),
         ("true_distance (2 stages)", "true_distance (20 stages)"),
         ("kociemba (2 stages)", "kociemba (20 stages)"),
         ("kociemba (2 stages)", "true_distance (2 stages)"),
         ("mixed", "true_distance (2 stages)"), ("mixed", "kociemba (2 stages)")]


def compare_rows(g):
    rows = []
    for a, b in PAIRS:
        if a not in g or b not in g:
            continue
        for title, metric, part, name, pct in COMPARE:
            va = [v for v in (val(r, name, metric, part) for r in g[a].values()) if v is not None]
            vb = [v for v in (val(r, name, metric, part) for r in g[b].values()) if v is not None]
            if len(va) < 2 or len(vb) < 2:
                continue
            d, lo, hi, z = welch_diff(va, vb)
            rows.append({"a": a, "b": b, "measure": title, "pct": pct, "diff": d, "lo": lo, "hi": hi, "z": z,
                         "excludes_0": not (lo <= 0 <= hi), "n": f"{len(va)}/{len(vb)}"})
    return rows


def compare_section(g):
    rows = compare_rows(g)
    if not rows:
        return []
    out = []
    for c in rows:
        f = (lambda x: f"{100 * x:+.1f} pts") if c["pct"] else (lambda x: f"{x:+.2f}")
        out.append([f"{c['a']} minus {c['b']}", c["measure"], f(c["diff"]), f"{f(c['lo'])} to {f(c['hi'])}",
                    f"{c['z']:.1f}", "yes" if c["excludes_0"] else "no", c["n"]])
    return (["### Key comparisons", "",
             "Difference of means over seeds; 95% Welch t-interval with each seed as one observation; "
             "'Diff/SE' is the difference divided by its standard error; 'Excludes 0' marks intervals "
             "that do not contain zero. The same test cubes are used for every run.", ""]
            + md_table(["Comparison", "Measure", "Difference", "95% interval", "Diff/SE", "Excludes 0", "Seeds"],
                       out))


def sweep_section(sg, g):
    md = []
    widths = sorted({int(w) for runs in sg.values() for r in runs.values()
                     for s in r["sets"].values() for w in s})
    for metric, title in (("solve_rate", "solve rate"), ("mean_length_solved", "mean solution length (solved)")):
        for name in DEEP:
            head = ["Arm", "greedy"] + [f"width {w}" for w in widths] + ["width 4096"]
            rows = []
            for a, runs in sg.items():
                row = [f"{a} (n={len(runs)})",
                       fmt([val(r, name, metric, "greedy") for r in g.get(a, {}).values()], metric in PCT)]
                for w in widths:
                    row.append(fmt([r["sets"].get(name, {}).get(str(w), {}).get(metric) for r in runs.values()],
                                   metric in PCT))
                row.append(fmt([val(r, name, metric, "beam") for r in g.get(a, {}).values()], metric in PCT))
                rows.append(row)
            md += [f"#### Beam {title}, {name}", ""] + md_table(head, rows)
    return md


def timing_section(kind):
    tdir = RES / "final_eval" / "timing"
    if not tdir.exists():
        return []
    times = {}
    for p in sorted(tdir.glob("final_*.json")):
        m = re.match(r"^final_(\w+?)_s(\d+)(_bestval)?\.json$", p.name)
        if kind == "final" and m.group(3):
            continue
        if kind == "bestval" and not m.group(3) and (tdir / p.name.replace(".json", "_bestval.json")).exists():
            continue
        times.setdefault(m.group(1), []).append(json.loads(p.read_text()))
    md = []
    for key, title, digits in (("beam_seconds_per_cube", "Beam (4096) seconds per cube, clean timing", 3),
                               ("greedy_seconds_per_cube", "Greedy seconds per cube, clean timing", 4)):
        head = ["Set", "cubes timed"] + [f"{a} (n={len(times.get(a, []))})" for a in ARMS]
        rows = []
        for name in SETS:
            n = next((t["sets"][name]["n"] for a in ARMS for t in times.get(a, []) if name in t["sets"]), "-")
            rows.append([name, str(n)] + [fmt([t["sets"][name][key] for t in times.get(a, [])], False, digits)
                                          for a in ARMS])
        md += [f"### {title}", "", "Seeds 0 to 2 of the 20-stage curricula and mixed; one process per H200, "
               "same batching as the evaluation (greedy: the whole subsample at once; beam: 32 cubes at a "
               "time); wall time / cubes.", ""] + md_table(head, rows)
    return md


# ---------- tidy CSV ----------
def tidy_rows():
    rows = []
    for kind in ("final", "bestval"):
        for label, runs in groups(kind).items():
            for seed, r in sorted(runs.items()):
                for name, e in r["sets"].items():
                    base = [label, seed, kind, r["step"], name]
                    if e.get("optimal_move_rate") is not None:
                        rows.append(base + ["one_move", "", "optimal_move_rate", e["optimal_move_rate"]])
                    for part in ("greedy", "beam"):
                        w = "" if part == "greedy" else r.get("beam_width", 4096)
                        for mtr in ROLL_METRICS:
                            if e[part].get(mtr) is not None:
                                rows.append(base + [part, w, mtr, e[part][mtr]])
        for label, runs in sweep_groups(kind).items():
            for seed, r in sorted(runs.items()):
                for name, ws in r["sets"].items():
                    for w, e in ws.items():
                        for mtr in ROLL_METRICS:
                            if e.get(mtr) is not None:
                                rows.append([label, seed, kind, r["step"], name, "beam", int(w), mtr, e[mtr]])
    return rows


def main():
    base = baselines()
    md = ["# Results summary", "",
          "Made by `scripts/summarize_eval.py`. Mean over seeds per arm, range over seeds in brackets. "
          "Rollouts use the baselines' fixed subsample of at most 1,000 cubes per set; the optimal-move rate "
          "uses every cube with known optimal moves. d1_4_all is in the training pool by design.", ""]
    for kind, title in (("final", "Primary: step-100,000 checkpoints"),
                        ("bestval", "Secondary: best-validation checkpoints")):
        g = groups(kind)
        steps = {a: {s: r["step"] for s, r in sorted(runs.items())} for a, runs in g.items()}
        md += [f"## {title}", "", "Checkpoint steps: " + json.dumps(steps), ""]
        md += compare_section(g) + eval_section(g, base)
        sg = sweep_groups(kind)
        if sg:
            md += ["### Beam-width sweep", ""] + sweep_section(sg, g)
        md += timing_section(kind)
    (RES / "summary.md").write_text("\n".join(md))
    rows = tidy_rows()
    with open(RES / "eval_tidy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "checkpoint", "step", "set", "method", "beam_width", "metric", "value"])
        w.writerows(rows)
    print(f"wrote results/summary.md ({len(md)} lines) and results/eval_tidy.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
