# Decisions made during implementation

Changes and gate outcomes relative to `Implementation plan.md`.

## 2026-11-02

**Layout.** Code and git repo in the `AI project` folder; Python environment, data, checkpoints and runs in `D:\rubiks-cube` (`rubik/paths.py`). GitHub repo name and visibility not decided yet (local git only).

**Exact table depth: 6** (decision 11 kept). Built in 12.5 s; counts match the known figures. Labels are exact one move deeper (depth 7) via neighbor lookup. About 90% of states at every depth have exactly one optimal move (mean about 1.1).

**Hold-out: 5% of depth 5 and 6** (Gate C item settled early, since it must be fixed before training). 28,745 depth-5 and 380,922 depth-6 states, seed 5, frozen in `holdout_d5_d6.pt`.

### Gate A
Measurements (`scripts/speed_kociemba.py`, RTX 3060 Laptop, 8 CPU threads):
- Kociemba (`kociemba` package, max_depth 24): about 16 ms median per random cube on one core; 195 solves/s on random cubes and about 3,000/s on 10-move scrambles with 8 warm workers. A max_depth below a cube's reachable length makes it search forever, so the default of 24 is always used.
- Exact GPU forward search into the table (`rubik/exact_search.py`): about 0.04 s per cube at distance 8 to 10, 0.5 s at 12, 5 s at 13; nothing beyond 13. Peak GPU memory 3.1 GB.

Decisions:
- **True-distance cap: 13.** Exact distances up to 13 from the table plus forward search; deeper cubes are marked as Kociemba estimates.
- **Decision 13 replaced:** depth 0 to 7 labeled exactly on the fly from the table. Depth 8+ uses a **precomputed pool with path labeling**: each Kociemba solution also labels every cube along its path (next move, remaining length), with exact labels for distances 8 to 12 where computed. The same pool is used by all arms. Path labels are correlated and of Kociemba quality, which the report must state.
- **Optimal solver for uniformly random cubes: skipped.** Random cubes get Kociemba length only; decision 21's check of true distance on random cubes is dropped, and the report states this.

### Gate B
`scripts/gate_b_check.py`, 1,260 scrambles of length 7 to 13, all with exact distances:
- Mean Kociemba length minus true distance: 1.9 (length 7) rising to 6.5 (length 12); Kociemba is exactly right in 71% of cubes at length 7 and 15% at length 13.
- Spearman rank correlation between true distance and Kociemba length: 0.72. A Kociemba length of 14 to 17 covers true distances 7 to 13.
- **Outcome: the arms differ strongly; both kept.** "Kociemba length" means the `kociemba` package output with default settings (first solution of 24 moves or fewer), a weak two-phase estimate. The report states this.

## 2026-11-03

**Baselines** (`results/baselines.json`): random moves pick an optimal move at chance (about 6%) and solve nothing; the 2-move lookahead solves 13% of the depth 1 to 4 set and almost nothing deeper; Kociemba's first move is optimal for about 90% of cubes up to depth 9, 77% at 10 and 27% at 13.

**Batched exact labeler** (`rubik/batch_label.py`): exact labels to distance 8 at about 50,000 cubes/s, to 9 at about 4,300/s, to 10 at about 233/s.

**Training targets: the same for all arms.** Every cube has one target: its exact optimal moves where known, otherwise the Kociemba move. Arms differ only in stage order (true distance, Kociemba length, or none). With a cumulative curriculum all arms have seen the same full pool by the last stage.

**Training pool**:
- Path pool, about 20M cubes: start cubes are half uniformly random, half scrambles of length 8 to 20; every cube along each Kociemba solution is labeled with the next move and the remaining length. Cubes at distance 7 or less (labeled exactly on the fly from the table) and all blocked test cubes are excluded. Exact labels up to distance 9 for every cube; the rest are flagged as distance 10 or more.
- Exact pools: about 300,000 cubes at distance 10, 30,000 at 11 and 10,000 at 12.
- Correction: exact labels at distance 10 for the whole path pool would take about a day (233 cubes/s), so distance 10 to 12 come from the dedicated exact pools instead.

**True-distance arm ordering:** exact distance up to 12; beyond that, Kociemba estimate. Tests stay exact up to 13.

**Kociemba lengths for depth 1 to 7 training cubes are precomputed.** Kociemba length differs from true distance for 6% of depth-3 cubes, rising to 24% at depth 6 and 31% at depth 7 (up to 10 moves off), so the Kociemba arm must order shallow cubes by their own Kociemba length. All depth 1 to 6 table cubes (hold-out excluded) and a 2M-cube depth-7 pool (depth-7 test cubes excluded) get Kociemba labels. All training data is therefore one precomputed pool where every cube has its exact label (where known), target and Kociemba length.

**Equal-share stages.** All arms get the same number of stages, the same steps per stage and the same total steps. Each stage adds the same share of the pool, cut at the arm's own difficulty quantiles (true distance or Kociemba length); the mixed control samples the whole pool from the start. Only the order of cubes differs between arms. The number of stages is set at Gate C.

**Engineering:** Kociemba workers run as plain subprocesses (`python -m rubik.koc_worker`). With `multiprocessing` on Windows, each worker re-runs the main script and imports torch (1 to 2 GB each), which exhausted memory twice.

**Pool built:** 30.4M cubes in total (20.2M path, 340,000 exact at distance 10 to 12, 9.8M depth 1 to 7 with Kociemba lengths); 0.5% held back for validation. Build time about 6 hours on the laptop.

**Gate C pilot** (default model 1024x4, 8.9M parameters; 40,000 steps, 10 stages, batch 4096, one seed; 26 minutes per arm at about 25 steps/s on the laptop GPU). Validation share, final step:

| Arm | Optimal move, exact cubes | Depth 8-9 | Depth 10-12 | Depth 13+ (Kociemba target) | Train hit / val hit |
|---|---|---|---|---|---|
| True distance | 87.3% | 70.0% | 65.7% | 45.5% | 70.8% / 67.6% |
| Kociemba | 87.1% | 71.1% | 66.3% | 47.2% | 72.0% / 68.3% |
| Mixed | 87.9% | 71.6% | 68.0% | 54.7% | 75.2% / 71.9% |

- Mixed is ahead throughout, most at depth 13+, where curriculum arms meet those cubes only in the last stages.
- Train and validation hit rates are close, so the model underfits rather than overfits: more capacity or more steps may help. To be tested with the wide model before the final runs.
- One seed; differences of about 1 point are within noise.

**Wide model check** (2048x8, 68M parameters, mixed arm, same 40,000-step schedule, stopped at step 10,000; 4.6 steps/s on the laptop, 5.5 times slower than the default). Validation share at step 10,000, wide vs default: optimal move on exact cubes 84.2% vs 81.4%, depth 8-9 61.6% vs 56.8%, depth 10-12 64.6% vs 64.2%, depth 13+ 53.1% vs 51.7%, train hit 70.9% vs 68.5%. The wide model is ahead by 1 to 5 points and still underfits. A full-length comparison is left for the GPU machine, where the wide model is cheap. Pilot logs are in `results/pilot/`.

**Data on the GPU machine is rebuilt from scratch** (user decision), with `scripts/build_all.sh`; `scripts/check_data.py` confirms the table, hold-out and test sets match the recorded checksums (all match on the laptop). The cube20.org positions file moved into the repo (`resources/`).
