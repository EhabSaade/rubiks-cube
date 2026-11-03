# Project context (handoff, 2026-11-03)

Everything a new session needs to continue. The detailed record of every decision and measurement is [docs/decisions.md](docs/decisions.md); this file is the overview and the current state.

## The project
Solo project, planned for 4 to 8 weeks (about 6), first code commit 2026-11-02. Research question (fixed, from the proposal): **to what extent can a neural network predict strong or optimal next moves on a 3x3 Rubik's cube, and does curriculum learning help?**

- Half-turn metric, 18 moves.
- Policy-only residual MLP; input is a 480-value cubie one-hot encoding. No symmetry augmentation.
- Three training arms:
  - **true_distance:** curriculum ordered by exact distance.
  - **kociemba:** curriculum ordered by Kociemba solution length.
  - **mixed:** no curriculum; the whole pool from the start.
- 3 seeds per arm, so 9 runs. Results are reported descriptively, with no pass or fail rule.
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
| 4 | Model, training loop, pool, pilot; Gate C | Pilot done; **Gate C decisions open** |
| 5 | 9 full runs, then evaluate all on the frozen test sets | **Next** |
| 6 | Charts by depth per arm with all seeds, report, single rerun command | Later |

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

## Data (not in git)
All large files live under the storage folder: `D:\rubiks-cube` on the laptop, or the `RUBIK_STORAGE` environment variable.
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
- **Transfer bundle:** `D:\rubiks-cube\rubik_data.tar` (2.55 GB), with sha256 `8058f915fa05acc42e4a72fc39be85a35a538dedd7dd8fe22f8a0850fb027565` in `rubik_data.tar.sha256`. It contains all of the above.
  - Rebuilding from scratch is possible (see README) but takes about 8 hours and needs kociemba. Copy the tar instead.

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

## Open decisions (the user decides; bring evidence and a recommendation)
1. **Gate C:**
   - Model size: default 1024x4 or wide 2048x8 (`configs/wide.yaml`).
   - Total steps and number of stages; the pilot used 40k steps and 10 stages, the config default is 100k steps.
   - Beam width.
   - Suggested first job on the GPU machine: a short speed test, then a full-length wide vs default comparison on one arm.
2. **The pilot finding:** the mixed arm beats both curricula. That is a legitimate result for the research question. Any change to the curriculum design (for example, mixing in some deep cubes in early stages) is a method change and needs the user's approval and a log entry.
3. **Metric list, before final testing.** Suggested:
   - Optimal-move rate.
   - Distance-reduction rate.
   - Solve rate by depth (greedy and beam).
   - Solution length.
   - Solve time.
   - Heuristic progress on unsolved cubes: misplaced or misoriented cubies, with correct stickers as a secondary measure.

## Next steps
1. Set up the GPU machine (see AGENTS.md) and verify the data checksum and the tests.
2. Run a speed test there for both model sizes, a few hundred steps each.
3. Bring the Gate C decision to the user and log it in `docs/decisions.md`.
4. Run the 9 final runs with `scripts/run_arms.sh`.
5. Write the final evaluation script, once the user has confirmed the metrics:
   - It loads each run's final checkpoint and evaluates on every frozen test set.
   - Use greedy and beam rollouts (`rubik/evaluate.py`: `greedy_rollout`, `beam_rollout`, `model_policy`) and progress measures.
   - Report per depth.
   - Save summaries in `results/`.
   - Baselines must use the same sets: see `scripts/run_baselines.py`, which subsamples 1,000 cubes per set for rollouts.
6. Charts and report (Step 6).

## Known engineering lessons
- **Kociemba workers run as plain subprocesses** (`python -m rubik.koc_worker`), never `multiprocessing`. On Windows, spawn re-imports torch in every worker and exhausted memory twice.
- **Kociemba with max_depth below what a cube needs can hang** (superflip at 18). Always use 24.
- **Laptop memory:** run only one heavy torch process at a time on the 6 GB GPU. On the GPU machine, run one training run per GPU.
- **`HoldoutFilter.contains` is not chunked:** pass chunks of at most about 500k cubes.
- **The editor once saved a stray "ok" into `rubik/scramble.py`.** If imports fail with a SyntaxError, check `git diff` first.
