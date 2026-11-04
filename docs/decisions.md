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

### GPU machine
**Machine:** Slurm job on node galileo4 (partition public, job ends 2026-11-07 23:29): 20 CPU cores of the node's 224, 4 NVIDIA H200 NVL (141 GB each), Ubuntu 22.04, Python 3.10.12. Storage folder `~/rubiks-cube`. `scripts/cloud_setup.sh` installed everything; 35 tests passed.

**Fresh data build** (`RUBIK_WORKERS=20`): table 2.7 s; test sets 11 minutes (laptop about 2 hours; d13 alone 503 s, at about 2 cubes/s, against 5 s per cube on the laptop). `check_data.py`: the table, the hold-out and all 13 test sets match the recorded checksums, so the baselines and pilot results stay valid. The pool part of `build_all.sh` is CPU-bound (20 Kociemba workers on 20 cores, GPU idle), so after 6 path chunks the three pool parts were restarted to run at the same time (path and shallow on GPU 0, exact on GPU 1, 20 workers each). Each part is seeded on its own, so the pool is the same as from a sequential build. Times: shallow 21.5 minutes, exact 26 minutes, path POOL_PATH_TIME; pool POOL_TOTAL cubes.

**Speed test on the H200** (mixed arm, 400 steps, while the build used all CPU cores, so lower bounds): default 1024x4 103 steps/s, wide 2048x8 61 steps/s (laptop 25 and 4.6). With 3 runs on one GPU: about 65 steps/s each for the default, about 27 each for the wide model (GPU 99% busy).

**Beam timing** (untrained model on fresh random cubes, so no cube is solved early and every rollout runs 100 moves; seconds per cube): default model: width 16 0.35, 256 0.42, 1024 0.43, 4096 0.49; wide model: 0.42, 0.44, 0.60, 1.55. Greedy rollouts of 1,000 cubes: 0.45 s (default) and 0.60 s (wide). Up to width 1024 the cost is set by the 100 sequential steps, not by the width.

### Gate C
User decisions:
- **Model: wide 2048x8** (68M parameters, `configs/wide.yaml`).
- **100,000 steps per run.**
- **20 equal-share stages** of 5,000 steps; the curriculum arms train on the full pool for the last 5,000 steps.
- **Beam width 4096** for the final evaluation; greedy rollouts are reported as well.
- Unchanged from the config: batch 4096, learning rate 0.001 with 500 warm-up steps and cosine decay, weight decay 0.0001, gradient clip 1.0.

**Metric list for the final evaluation** (user decision; `scripts/evaluate_runs.py`), per run and test set, step-100,000 checkpoint:
- Optimal-move rate: the top move is optimal, over every cube with known optimal moves.
- Greedy and beam (width 4096) rollouts, 100-move cap, visited states forbidden, on the same subsample of at most 1,000 cubes per set as the baselines (`run_baselines.rollout_indices`): solve rate, also by exact distance within d1_4_all; mean solution length; mean excess over the exact distance and over the Kociemba length (solved cubes); seconds per cube.
- Progress on unsolved cubes: misplaced, misoriented, and misplaced-or-misoriented pieces (of 20); correct stickers (of 48) as a secondary measure.
- The distance-reduction rate is dropped: in the half-turn metric a move lowers the exact distance exactly when it is optimal, so it equals the optimal-move rate.

**Batched beam search** (user approved; `rubik/evaluate.py`: `beam_rollout_batched`): the same search as `beam_rollout`, run for 32 cubes at once, with history and duplicates matched on exact (cube, hash) pairs. Checked identical to `beam_rollout` (solved, length and final state) in `tests/test_evaluate.py`, and at width 4096 with a wide checkpoint (mixed seed 1, step 44,000) on 24 fresh cubes (scrambles of length 8, 11, 14 and random states; no test set): identical for 24 of 24, 11.0 s against 19.5 s, while two training runs shared the GPU. At width 4096 most of the time is the model's float32 arithmetic, so batching saves less there than at small widths. The step-44,000 checkpoint solved all 24 fresh cubes with beam 4096, the random ones in 20 to 22 moves.

**Overfitting of the wide model on the mixed arm** (validation share, final runs). Mixed peaks at a validation target-hit rate of 72.6 to 73.1% around steps 30,000 to 50,000 and falls to 71.5 to 71.9% at 100,000, while its training hit rate reaches 95 to 96%; at depth 13+ it peaks at about 54.5% (step 30,000) and falls to about 51%. The curriculum arms are at their best at the end (67.2 to 67.6%), since they see the full pool only in the last 5,000 steps. The test-set evaluation was paused when this was seen (only d1_4_all had been reported, 100% for every measure) and resumed after the decision below.

**Checkpoints for the final evaluation** (user decision, fixed before any test result): primary, the step-100,000 checkpoint of every run, as decided at Gate C; secondary, per run the kept checkpoint (every 10,000 steps) with the highest validation target-hit rate, ties to the later step (`scripts/evaluate_runs.py --checkpoint best_val`). Selected so far: mixed seed 0 step 50,000 (73.05% against 71.86% at 100,000), mixed seed 1 step 40,000 (73.08% against 71.55%); every finished curriculum run at step 100,000, so its secondary result is its primary one.

**Final runs trained** (wide 2048x8, 100,000 steps, 20 stages, batch 4096, seeds 0 to 2): all 9 at once on the 4 GPUs (3 on GPU 0, 2 on each of the others), 21:09 to 22:12. Each took 43 to 45 minutes at about 40 steps/s with 2 runs on a GPU, and 62 to 63 minutes at about 28 steps/s with 3. Logs are in `results/final/`. Validation share, step 100,000:

| Arm | Seed | Val hit | Optimal move, exact cubes | Depth 8-9 | Depth 10-12 | Depth 13+ | Train hit | Best kept checkpoint |
|---|---|---|---|---|---|---|---|---|
| true_distance | 0 | 67.2% | 88.3% | 69.8% | 65.2% | 43.8% | 77.7% | 67.2% (100k) |
| true_distance | 1 | 67.2% | 88.6% | 70.3% | 64.7% | 43.7% | 77.6% | 67.2% (100k) |
| true_distance | 2 | 67.6% | 88.7% | 71.2% | 64.8% | 44.3% | 77.7% | 67.6% (100k) |
| kociemba | 0 | 67.5% | 88.3% | 72.3% | 66.1% | 44.0% | 79.1% | 67.5% (100k) |
| kociemba | 1 | 67.8% | 88.5% | 71.2% | 65.6% | 44.9% | 79.1% | 67.9% (90k) |
| kociemba | 2 | 67.3% | 87.9% | 69.9% | 65.8% | 44.3% | 78.8% | 67.3% (100k) |
| mixed | 0 | 71.9% | 90.5% | 79.2% | 69.2% | 51.0% | 95.9% | 73.0% (50k) |
| mixed | 1 | 71.5% | 90.2% | 78.1% | 68.9% | 50.8% | 95.4% | 73.1% (40k) |
| mixed | 2 | 71.1% | 89.8% | 76.8% | 68.2% | 50.3% | 95.7% | 72.6% (40k) |

- Seeds agree within about 1 point; mixed is 4 points ahead of both curricula at step 100,000 and 5 to 6 points ahead at its best checkpoint.
- Against the pilot (default model, 40,000 steps, 10 stages): the curricula are about the same overall (67.2 to 67.8% against 67.6 and 68.3%) and at depth 13+ slightly lower (43.7 to 44.9% against 45.5 and 47.2%); mixed is the same at step 100,000 (71.1 to 71.9% against 71.9%) and about 1 point higher at its best checkpoint.
- Best-validation checkpoints for the secondary evaluation: mixed seeds 0, 1, 2 at steps 50,000, 40,000, 40,000; kociemba seed 1 at step 90,000; the other curriculum runs at step 100,000.

**Final evaluation done** (`scripts/evaluate_runs.py`, beam width 4096, 21:52 to 23:18; all tables in `results/final_eval/summary.md`, made by `scripts/summarize_eval.py`; per-run files in `results/final_eval/`). Means over 3 seeds; primary = step 100,000, with the secondary (best-validation) value in brackets where it differs:

| | true_distance | kociemba | mixed | Kociemba package |
|---|---|---|---|---|
| Beam solve rate, every set incl. random and distance 20 | 100% | 100% | 100% | 100% |
| Beam mean excess over exact distance, d13 | 4.43 | 4.73 (4.78) | 7.39 (4.71) | 6.40 |
| Beam mean length, random | 21.89 | 21.96 (21.98) | 22.70 (21.19) | 20.79 |
| Beam mean length, d20_known (optimal 20) | 23.00 | 23.03 (23.12) | 23.55 (21.94) | 21.55 |
| Greedy solve rate, d10 | 47.9% | 50.5% (50.8%) | 81.8% (80.7%) | |
| Greedy solve rate, d13 | 30.3% | 35.5% (35.9%) | 78.6% (75.3%) | |
| Greedy solve rate, random | 28.5% | 32.0% (32.5%) | 74.4% (73.3%) | |
| Optimal-move rate, d7 | 90.7% | 90.0% (89.9%) | 91.1% (91.4%) | first move 90.4% |
| Optimal-move rate, d10 | 39.6% | 39.4% (39.3%) | 34.1% (38.5%) | first move 77.3% |
| Optimal-move rate, d13 | 19.8% | 19.0% (18.9%) | 15.3% (18.3%) | first move 26.7% |

- Up to depth 6 every arm picks an optimal move 99 to 100% of the time and solves every cube optimally, greedy and beam. d1_4_all is in the training pool (all depth 1 to 4 states), so its 100% is expected; the hold-out sets d5 and d6 were never trained on.
- Greedy rollouts: mixed solves far more deep cubes than both curricula (d13 78.6% against 30.3 and 35.5%; random 74.4% against 28.5 and 32.0%), at both checkpoints.
- Beam 4096 solves every test cube for every run, so the arms differ only in solution length there: at d10 to d13 the curricula and best-validation mixed are close (d13 excess 4.4 to 4.8), the overfitted step-100,000 mixed is worse (7.4); on random and distance-20 cubes best-validation mixed is shortest (21.19 and 21.94).
- The optimal-move rate on deep exact cubes is 3 to 5 points higher for the curricula than for step-100,000 mixed, and within 1 to 2 points of best-validation mixed, while mixed's greedy solve rate is more than twice theirs.
- Caveat on solve time: the evaluations ran 2 to 4 per GPU, so the seconds-per-cube figures depend on the sharing and are not clean timings: beam 0.02 to 0.8 s per cube, greedy under 0.01 s. Everything else is deterministic; the 2 kociemba seed 1 evaluations were moved to free GPUs and rerun from the start.

**Clean solve-time measurement** (user decision; `scripts/time_eval.py`, 23:23 to 23:39): the 13 distinct checkpoints (9 final, plus best-validation mixed seeds 0 to 2 and kociemba seed 1), one process per H200 with nothing else running, after a warm-up; per test set a fixed subsample of 200 cubes (every 5th of the baselines' 1,000-cube rollout subsample; all 11 of d20_known), the same batching as the evaluation (greedy: the whole subsample at once; beam: 32 cubes at a time); seconds per cube = wall time / cubes. Results in `results/final_eval/timing/`; the tables in `results/final_eval/summary.md` now use these and no longer the shared-GPU figures. Beam solved every timed cube; the greedy solve rates on the subsample are within sampling noise of the full evaluation (for example d13: 25.0 to 81.0% against 28.9 to 79.3%).
- Beam 4096, seconds per cube (means over seeds, primary checkpoints): 0.011 at d1_4, 0.051 at d7, 0.094 (curricula) and 0.122 (mixed) at d10, 0.19 and 0.23 at d13, 0.25 and 0.26 on random cubes, 0.27 and 0.28 on distance-20 cubes. The overfitted step-100,000 mixed is slower at d10 to d13 because its solutions are longer; best-validation mixed is about as fast as the curricula there and slightly faster on random cubes (0.239).
- Greedy, seconds per cube: about 0.002 for a batch of 200 for every arm and depth (0.03 for the 11 distance-20 cubes). The batch runs until its slowest cube is solved or reaches 100 moves, so the per-cube figure reflects batch size and the move cap more than the policy.
- Run-to-run spread on the same set is a few thousandths of a second, and all 4 GPUs give the same times.

## 2026-11-04

**Three more seeds per arm** (user decision, made after the 3-seed test results and fixed before any further run): seeds 3, 4 and 5 for all three arms, 9 runs with exactly the Gate C configuration, evaluated in the same way (primary step-100,000 and secondary best-validation checkpoints). All 6 seeds per arm are reported, whatever the new seeds show; the 3-seed results above stay on record as the planned result. Reason: the main result is already firm with 3 seeds (mixed against kociemba, greedy solve rate at d13: +43 points, 25 times the standard error over seeds), but the kociemba against true_distance ordering difference is not (+5.2 points at d13, 2.6 times the standard error); 6 seeds shrink the standard error by about the square root of 2.

**Integrity check: no test cube is in the training pool.** All 30,378,368 pool cubes (training and validation shares) against every frozen test set, by hash and then full state: 0 of the 418,678 cubes of holdout_d5, holdout_d6, d7, d8 to d13, random and d20_known occur in the pool; d1_4_all (46,740) is entirely in the pool, by design (in-distribution set).

**How different the two curricula are in the pool** (training share, 30,227,043 cubes): the exact distance is known for 45.0%; for the other 55.0% the true_distance key falls back on the Kociemba length, the same key as in the kociemba arm. Overall the two keys are equal for 90.2% of cubes. With 20 stages (seed 0 tie-breaking), 35.3% of cubes are in the same stage in both curricula, 72.4% within one stage; mean stage difference 1.47. So the two curricula differ much less in training than the Gate B correlation (0.72, on scrambles with exact distances 7 to 13) suggests, which fits their near-identical results; the report should state this.

**Two secondary analyses** (user decision, 2026-11-04, fixed before running; both reported in full, separately from the main comparison):
- **2-stage curricula:** both curriculum arms rerun with `n_stages=2` (the easier half of each arm's order for steps 0 to 49,999, then the full pool), seeds 0 to 5 (12 runs, run names `stages2_<arm>_s<seed>`), every other setting as at Gate C. It tests whether the main result depends on the curricula meeting the hardest cubes only at the end (20 stages: last 5% of steps). The mixed arm does not use stages, so its existing 6 runs are the comparison. Evaluated like the main runs (beam 4096, step-100,000 and best-validation checkpoints).
- **Beam-width sweep:** beam widths 2, 4, 16, 64, 256 and 1,024 (4,096 is in the main evaluation; width 1 equals greedy) on d8 to d13, random and d20_known, same 1,000-cube subsamples, for every primary checkpoint and every best-validation checkpoint that differs from it (`scripts/beam_sweep.py`). Reason: beam 4096 solves every test cube for every run, so only smaller widths can show how much search each arm needs.
  Widths 2 and 4 were added after a test of the script on one checkpoint (mixed seed 0, step 100,000) solved every d8 and d20_known cube already at width 16, so smaller widths are needed to see where search starts to matter; no sweep result had been produced otherwise.
- **Working without the user** (user instruction, 2026-11-04 night): the jobs run on a schedule and the user is not notified; decisions inside what the user has already decided (for example rerunning a failed job) are taken and logged here; any method change is only proposed, not run.
- The summary tables moved to `results/summary.md` (all groups, 6 seeds, 2-stage curricula, beam sweep), with every number also in `results/eval_tidy.csv`; key comparisons use a 95% Welch t-interval over seeds (each seed one observation; the same test cubes for every run). A percentile bootstrap was tried first and dropped: with 3 seeds it gives intervals that are too narrow (kociemba minus true_distance, greedy d13: bootstrap +2.1 to +8.3 points, Welch -1.2 to +11.6).

**Seeds 3 to 5 trained and evaluated** (Gate C configuration; training 00:14 to 01:16, 2 runs per GPU and 3 on one (44 and 62 minutes); evaluation 00:58 to 02:30, one evaluation per GPU at a time, both checkpoints; logs in `results/final/`, per-run files in `results/final_eval/`). Validation share, step 100,000:

| Arm | Seed | Val hit | Optimal move, exact cubes | Depth 8-9 | Depth 10-12 | Depth 13+ | Train hit | Best kept checkpoint |
|---|---|---|---|---|---|---|---|---|
| true_distance | 3 | 67.3% | 88.3% | 70.0% | 64.8% | 44.2% | 77.5% | 67.3% (90k) |
| true_distance | 4 | 67.2% | 88.1% | 69.2% | 65.5% | 43.8% | 77.5% | 67.3% (90k) |
| true_distance | 5 | 66.7% | 87.8% | 68.0% | 65.4% | 43.1% | 77.7% | 66.8% (90k) |
| kociemba | 3 | 67.4% | 87.9% | 70.0% | 65.5% | 44.6% | 78.9% | 67.4% (100k) |
| kociemba | 4 | 67.4% | 88.1% | 71.2% | 65.8% | 44.1% | 79.0% | 67.4% (90k) |
| kociemba | 5 | 67.3% | 88.1% | 71.6% | 65.1% | 44.1% | 78.8% | 67.3% (100k) |
| mixed | 3 | 71.4% | 90.4% | 78.4% | 69.3% | 50.0% | 95.9% | 72.8% (40k) |
| mixed | 4 | 71.5% | 90.2% | 77.5% | 68.3% | 51.0% | 95.6% | 72.8% (40k) |
| mixed | 5 | 71.4% | 90.1% | 77.6% | 67.9% | 50.9% | 95.6% | 72.6% (50k) |

- The new seeds repeat seeds 0 to 2 within about 1 point on validation; mixed again overfits (train hit 96% against validation 71%) and peaks at step 40,000 to 50,000. Best-validation checkpoints: true_distance seeds 3 to 5 at step 90,000 and kociemba seed 4 at step 90,000 (each 0.0 to 0.1 points above step 100,000), mixed at 40,000, 40,000, 50,000.

Test sets, means over all 6 seeds (seeds 0 to 2 | seeds 3 to 5 in brackets); primary = step 100,000, secondary (best-validation) value after the slash:

| | true_distance | kociemba | mixed |
|---|---|---|---|
| Beam mean excess over exact distance, d13 | 4.51 [4.43, 4.59] / 4.55 | 4.74 [4.73, 4.75] / 4.79 | 7.38 [7.39, 7.37] / 4.70 |
| Beam mean length, random | 21.89 [21.89, 21.89] / 21.92 | 21.95 [21.96, 21.94] / 21.97 | 22.71 [22.70, 22.71] / 21.20 |
| Beam mean length, d20_known | 23.02 [23.00, 23.03] / 22.98 | 23.02 [23.03, 23.00] / 23.11 | 23.53 [23.55, 23.52] / 21.88 |
| Greedy solve rate, d10 | 48.1% [47.9, 48.3] / 47.8% | 51.0% [50.5, 51.4] / 50.6% | 82.5% [81.8, 83.2] / 79.9% |
| Greedy solve rate, d13 | 30.7% [30.3, 31.1] / 30.5% | 35.8% [35.5, 36.2] / 35.6% | 78.2% [78.6, 77.8] / 74.1% |
| Greedy solve rate, random | 29.7% [28.5, 31.0] / 28.5% | 33.2% [32.0, 34.5] / 33.7% | 75.0% [74.4, 75.5] / 71.4% |
| Optimal-move rate, d7 | 90.8% [90.7, 91.0] / 90.9% | 90.0% [90.0, 90.0] / 89.9% | 90.8% [91.1, 90.5] / 91.3% |
| Optimal-move rate, d10 | 39.5% [39.6, 39.5] / 39.4% | 39.6% [39.4, 39.9] / 39.6% | 35.0% [34.1, 36.0] / 38.6% |
| Optimal-move rate, d13 | 19.4% [19.8, 19.0] / 19.2% | 18.9% [19.0, 18.9] / 18.9% | 15.3% [15.3, 15.3] / 17.9% |

- Beam 4096 again solves every test cube for every run. Every 3-seed conclusion holds, and no 6-seed mean moved by more than 1.2 points or 0.08 moves.
- 95% Welch intervals over 6 seeds (primary checkpoints; all in `results/summary.md`): mixed minus kociemba, greedy d13 +42.4 points (+39.7 to +45.0); mixed minus true_distance +47.5 (+45.2 to +49.8).
- Kociemba against true_distance, the comparison the extra seeds were for: kociemba's greedy solve rate is higher, d10 +2.9 points (+1.0 to +4.8), d13 +5.1 (+2.2 to +8.0; 3 seeds: -1.2 to +11.6), random +3.5 (+0.5 to +6.5); the same at the best-validation checkpoints. Its optimal-move rate is the same (d10 +0.1, -0.9 to +1.1; d13 -0.5, -1.9 to +0.9), and its beam solutions are slightly longer at d13 (excess +0.23 moves, +0.10 to +0.35), with no difference on distance-20 cubes (+0.00, -0.39 to +0.39). These intervals are not corrected for the several comparisons made; the greedy difference is small next to the mixed effect (about a tenth of it) and comes with near-identical training data (90% of the pool has the same key in both arms, see above).

**GPU node lost overnight; 2-stage runs resumed on a new node.** The Slurm job on galileo4 ended at about 03:08 and killed every process: the 12 two-stage runs had reached steps 84,000 to 88,000 (60,000 to 62,800 for the 3 runs started last), and no evaluation or sweep had started. Nothing ran from 03:08 to 10:50. The work continues in a new job on nlp-h200-1 (4x H200 NVL, same shared home folder, so data, runs and checkpoints are unchanged). Every run resumes from its `last.pt` (saved every 2,000 steps, with model, optimizer, schedule, sampler and random-generator states), so at most 1,800 steps per run are repeated. Each `log.jsonl` was first cut back to the resume step so that no step is logged twice; the uncut logs are kept in `$RUBIK_STORAGE/runs/stages2_logs_before_resume/`. Resuming had never been used before and failed at first: the checkpoint is loaded onto the GPU, and the random-generator states must be on the CPU (`TypeError: RNG state must be a torch.ByteTensor`). Fixed with `.cpu()` in `scripts/train.py` and `rubik/pool.py`, checked on a real checkpoint (the restored generator gives the same draws), and the queue was restarted (`$RUBIK_STORAGE/runs/phase2b.sh`: training, then the same task queue as before).
- Two queue workers per GPU from 11:07 (user: "keep pushing with the full utilization possible"; `$RUBIK_STORAGE/runs/phase2b_helper.sh`, the same task list and locks), so each GPU runs 2 evaluation or sweep jobs at once. Results do not depend on this (beam search is deterministic, cubes never interact). The seconds-per-cube and seconds fields in `results/stages2_eval/` and `results/beam_sweep/` are therefore shared-GPU times, like those in the first per-run evaluation files; only `results/final_eval/timing/` holds clean times.

**How often each arm shows the network a cube** (arithmetic from `CurriculumSampler`: 100,000 steps of 4,096 cubes over the 30,227,043 training cubes; expected draws per cube). Mixed: 13.6 for every cube. 20 stages: 48.8 for the easiest 5% of the arm's order, 2.9, 2.1, 1.4 and 0.7 for the four hardest 5% slices. 2 stages: 20.3 for the easier half, 6.8 for the harder half. The hardest fifth of each curriculum's order holds cubes with keys (Kociemba lengths) 17 to 23, of which 0.0% (true_distance) and 3.2% (kociemba) have an exact distance, against 45% in the whole pool (seed 0 tie-breaking). So with 20 stages this fifth is drawn about 0.7 to 2.9 times per cube against 13.6 in the mixed arm; the 2-stage runs raise this to 6.8.
