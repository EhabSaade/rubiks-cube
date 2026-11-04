# Project context (handoff, 2026-11-04)

Everything a new session needs to continue. The detailed record of every decision and measurement is [docs/decisions.md](docs/decisions.md); this file is the overview and the current state.

## The project
Solo project, planned for 4 to 8 weeks (about 6), first code commit 2026-11-02. Research question (fixed, from the proposal): **to what extent can a neural network predict strong or optimal next moves on a 3x3 Rubik's cube, and does curriculum learning help?**

- Half-turn metric, 18 moves.
- Policy-only residual MLP; input is a 480-value cubie one-hot encoding. No symmetry augmentation.
- Three training arms:
  - **true_distance:** curriculum ordered by exact distance.
  - **kociemba:** curriculum ordered by Kociemba solution length.
  - **mixed:** no curriculum; the whole pool from the start.
- 3 seeds per arm, so 9 runs (6 seeds, 18 runs since 2026-11-04, user decision). Results are reported descriptively, with no pass or fail rule.
- Solving is reported with greedy rollouts and beam search:
  - 100-move cap.
  - States already visited are forbidden; the next-best move is taken instead.
- Baselines: random moves, shallow search, Kociemba reference.
- Deliverables: a written report and reproducible code. The proposal PDF and `Implementation plan.md` are the user's local files, never committed.

## Plan steps and status
| Step | Content | Status |
|---|---|---|
| 1 | Setup, batched simulator, scrambles, encoding | Done |
| 2 | Exact table to depth 6, hold-out, Kociemba wrapper, speed test; Gates A and B | Done |
| 3 | Frozen test sets, rollouts, progress measures, baselines | Done |
| 4 | Model, training loop, pool, pilot; Gate C | Done (GPU machine, 2026-11-03) |
| 5 | 9 full runs, then evaluate all on the frozen test sets | Done (2026-11-03); 3 more seeds per arm (18 runs) done 2026-11-04 |
| 6 | Charts by depth per arm with all seeds, report, single rerun command | Done 2026-11-04: `scripts/reproduce_all.sh`, 2-stage curricula (12 runs), beam-width sweep (seeds 0 to 2 only, machine released), report |

## Gates and decisions so far
- **Gate A (speed test):**
  - True distance is exact up to 13: the table gives depth 7, and exact GPU search goes further.
  - Kociemba is the `kociemba` package, with max_depth 24.
  - No separate optimal solver is used.
- **Gate B (do the arms differ):** yes. The Spearman correlation between true distance and Kociemba length is 0.72, so both curriculum arms are kept.
- **Hold-out:** 5% of depth 5 and 6 states, blocked from training.
- **Training targets are identical for all arms:**
  - Each cube's target is its exact optimal-move set where known, otherwise the Kociemba move.
  - Arms differ only in the order cubes are introduced.
- **Equal-share cumulative stages:**
  - Every arm gets the same number of stages and the same steps per stage.
  - Each stage adds the same share of the pool, cut at the arm's own difficulty quantiles.
  - The mixed arm samples the whole pool from step 0.
- **GitHub:** public repo https://github.com/EhabSaade/rubiks-cube, owned and pushed by the user.
- **Gate C (2026-11-03):** wide model 2048x8 (68M parameters, `configs/wide.yaml`), 100,000 steps, 20 stages, beam width 4096.
- **Final evaluation:** metrics as in `scripts/evaluate_runs.py` (distance-reduction rate dropped: it equals the optimal-move rate); batched beam search (`beam_rollout_batched`, verified identical to `beam_rollout`). Checkpoints: primary step 100,000; secondary the best-validation kept checkpoint (mixed overfits after about 40,000 steps).

## Data (not in git)
All large files live under the storage folder: `D:\rubiks-cube` on the laptop, or the `RUBIK_STORAGE` environment variable.

**On the GPU machine (Slurm node galileo4, 4x H200, storage `~/rubiks-cube`), all data was rebuilt from scratch on 2026-11-03 and every checksum matched.** The build uses `scripts/build_all.sh` (the user's choice). Every build is seeded, and `scripts/check_data.py` compares the rebuilt table, hold-out and test sets with the checksums recorded in `results/` (`table_d6.json`, `holdout_d5_d6.json`, `testsets_manifest.json`). On the laptop, all of them match. The 100 cube20.org distance-20 positions are in `resources/cube20_random100.txt`. The list below describes the laptop build; a fresh build should give the same.
- `data/table_d6.pt` (417 MB): every state to depth 6 (8,240,067) with distance and an 18-bit mask of all optimal moves. It gives exact labels to depth 7.
- `data/holdout_d5_d6.pt`: 28,745 depth-5 and 380,922 depth-6 cubes, blocked from training.
- `data/testsets/*.pt` (465,418 cubes, frozen). Checksums are in `results/testsets_manifest.json`.
  - d1_4_all
  - holdout_d5
  - holdout_d6
  - d7
  - d8 to d13 (exact distances)
  - random
  - d20_known: superflip and 10 positions from cube20.org
  - Never rebuild or overwrite these.
- `data/pool/` (2.0 GB, 30.4M cubes); 0.5% (by hash) is the validation share. It has three parts:
  - `path/`: 20.2M cubes from Kociemba solution paths, exact labels to distance 9.
  - `exact/`: 300k cubes at distance 10, 30k at 11 and 10k at 12.
  - `shallow/`: 9.8M depth 1 to 7 cubes with Kociemba lengths.
- A packed copy of the laptop data also exists (`D:\rubiks-cube\rubik_data.tar`, 2.55 GB; `scripts/pack_data.sh`). The user chose not to use it and to rebuild instead.

## Key results so far
- **Baselines** (`results/baselines.json`):
  - Random moves pick an optimal move about 6% of the time and solve nothing.
  - A 2-move lookahead solves 13% of the depth 1 to 4 set.
  - Kociemba's first move is optimal for about 90% of cubes up to depth 9, 77% at 10 and 27% at 13.
- **Pilot** (Gate C evidence): default model 1024x4 (8.9M parameters), 40k steps, 10 stages, batch 4096, seed 0. About 26 minutes per arm on the laptop RTX 3060 (6 GB).

  | Arm | Optimal move, exact cubes | Depth 8-9 | Depth 10-12 | Depth 13+ (Kociemba target) |
  |---|---|---|---|---|
  | true_distance | 87.3% | 70.0% | 65.7% | 45.5% |
  | kociemba | 87.1% | 71.1% | 66.3% | 47.2% |
  | mixed | 87.9% | 71.6% | 68.0% | 54.7% |

  - Mixed is ahead throughout, most at depth 13+.
  - Training and validation hit rates are close (about 71% vs 68%), so the model underfits.
- **Wide model check:** 2048x8, 68M parameters, mixed arm, the same 40k schedule stopped at step 10k.
  - It is 1 to 5 points ahead of the default model at the same step. For example, 84.2% vs 81.4% optimal on exact cubes.
  - It still underfits.
  - On the laptop it runs at 4.6 steps/s, against 25 for the default model.
- Logs for all pilot runs are in `results/pilot/`. The format is one JSON line per 200 steps; validation every 2,000 steps.
- **Final runs** (18, wide model, seeds 0 to 5; logs in `results/final/`): validation hit at step 100,000 is 66.7 to 67.8% for both curricula and 71.1 to 71.9% for mixed; mixed overfits (training hit 96%, validation best about 73% at steps 40,000 to 50,000).
- **Final evaluation** (all tables in `results/summary.md`, every number in `results/eval_tidy.csv`, made by `scripts/summarize_eval.py`; means over 6 seeds): beam 4096 solves every test cube for every run, including random and distance-20 cubes. Greedy: mixed solves far more deep cubes (d13 78.2% against 30.7% true_distance and 35.8% kociemba; 95% Welch interval for mixed minus kociemba +39.7 to +45.0 points). Kociemba beats true_distance on greedy solve rate by 3 to 5 points (d13 +2.2 to +8.0), with the same optimal-move rate and slightly longer beam solutions at d13 (+0.23 moves). Curricula have 4 points higher optimal-move rates on deep exact cubes than step-100,000 mixed and give shorter beam solutions at d10 to d13; best-validation mixed is close to them there and shortest on random cubes. Clean solve times: beam 4096 about 0.01 s per cube at depth 1 to 4 up to about 0.25 s on random cubes; greedy about 0.002 s per cube in batches of 200.
- **2-stage curricula** (`results/stages2/`, `results/stages2_eval/`): greedy d13 69.8% (true_distance) and 70.1% (kociemba) against 30.7% and 35.8% with 20 stages and 78.2% for mixed; most of the curricula's deficit goes with how rarely 20 stages show the hardest cubes (0.7 to 2.9 draws per cube against 6.8 with 2 stages and 13.6 for mixed).
- **Report:** a Claude Docs document, https://claude.ai/artifact/7jNVkCjFya6VJHWaKxVsGx.

## Open decisions (the user decides; bring evidence and a recommendation)
1. **Gate C** and **the metric list**: decided on 2026-11-03 (see "Gates and decisions so far" and `docs/decisions.md`).
2. **The mixed arm beats both curricula** in the pilot and in the final runs (greedy solve rate, validation hit). That is a legitimate result for the research question. Any change to the curriculum design (for example, mixing in some deep cubes in early stages) is a method change and needs the user's approval and a log entry.
3. **Solve time:** re-measured cleanly (one process per GPU, `scripts/time_eval.py`, `results/final_eval/timing/`); the seconds per cube in the per-run evaluation files were measured with GPUs shared and are superseded.

## Next steps
All steps are done (2026-11-04). The GPU node was released after the 2-stage curricula and a beam-width sweep of seeds 0 to 2; the rest of the sweep (seeds 3 to 5, best-validation and 2-stage checkpoints) was not run and can be run with `scripts/beam_sweep.py` (or `scripts/reproduce_all.sh`) if wanted. Checkpoints (100 GB) and data (2.4 GB) stay in `~/rubiks-cube` on the cluster's shared home folder. Open question for the user: how to present the primary and secondary checkpoints in the report.

## Known engineering lessons
- **Kociemba workers run as plain subprocesses** (`python -m rubik.koc_worker`), never `multiprocessing`. On Windows, spawn re-imports torch in every worker and exhausted memory twice.
- **Kociemba with max_depth below what a cube needs can hang** (superflip at 18). Always use 24.
- **Laptop memory:** run only one heavy torch process at a time on the 6 GB GPU. On the GPU machine, run one training run per GPU.
- **`HoldoutFilter.contains` is not chunked:** pass chunks of at most about 500k cubes.
- **Resuming a run from `last.pt`** failed until 2026-11-04 (random-generator states loaded onto the GPU); fixed in `scripts/train.py` and `rubik/pool.py`.
- **The editor once saved a stray "ok" into `rubik/scramble.py`.** If imports fail with a SyntaxError, check `git diff` first.
