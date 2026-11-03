# Instructions for AI coding agents

Read [CONTEXT.md](CONTEXT.md) first (project state and open decisions), then [docs/decisions.md](docs/decisions.md) (full decision log) and [README.md](README.md) (setup and commands).

## Rules from the user
- **The user makes the method decisions.** At every gate and whenever a design choice or problem comes up, bring the evidence and a recommendation, then ask. Don't change the research question, arms, targets, test sets or curriculum design on your own.
- **Log every decision and measurement** in `docs/decisions.md`, under a dated heading, in the existing style.
- **Git:**
  - Commit locally with the user's identity: `Ehab Saade <ehabigb@gmail.com>`.
  - **Never add `Co-Authored-By: Claude` or any AI attribution** to commits.
  - The user pushes to GitHub themselves. Don't push, create remotes or change repo settings.
- **Never commit** `AI proposal.pdf`, `Implementation plan.md`, data, checkpoints or runs; `.gitignore` covers them. Small result summaries go in `results/`.
- **The frozen test sets are never rebuilt, overwritten or used during training.** Monitoring uses only the pool's validation share. Evaluation on test sets happens only after training.
- **Explain plainly.** The user wants to understand what happens and why, with numbers rather than vague claims.
- **When a long job finishes and the user is away, notify them with a question** (for example AskUserQuestion), not only a chat message.

## Setting up the GPU machine
1. Clone https://github.com/EhabSaade/rubiks-cube.
2. Copy `rubik_data.tar` and `rubik_data.tar.sha256` from the laptop (`D:\rubiks-cube\`, 2.55 GB). They are not in git.
3. **Linux:** `bash scripts/cloud_setup.sh /path/to/rubik_data.tar`. This checks the checksum, extracts to `$RUBIK_STORAGE` (default `~/rubiks-cube`), creates a venv with PyTorch 2.6 (CUDA 12.4), and checks the GPU, the pool and the tests. Then:
   ```
   export RUBIK_STORAGE=~/rubiks-cube
   source $RUBIK_STORAGE/.venv/bin/activate
   ```
   **Windows:** follow the README setup, extract the tar into the storage folder, and set `RUBIK_STORAGE`.
4. `nvidia-smi` shows the GPUs and whether they are busy. Use only GPUs the user says are free.
5. Check: `python -m pytest -q tests` should report 35 passed (on a GPU machine). `test_koc_worker` needs kociemba, which is not needed for training.

## Commands
- One run: `python scripts/train.py --config configs/base.yaml --arm mixed --seed 0 --set prefix=final total_steps=100000`
  - `--set` overrides any config key. `stop_at=N` pauses early.
  - Rerunning the same command resumes from `checkpoints/<run>/last.pt`.
- All 9 runs on several GPUs: `GPUS="0 1 2" CONFIG=configs/wide.yaml bash scripts/run_arms.sh prefix=final total_steps=100000`
  - Runs are split round-robin over the GPUs.
  - Per-run output is in `$RUBIK_STORAGE/runs/launch_<arm>_s<seed>.out`.
  - Logs are in `runs/<run>/log.jsonl`.
- Speed test: add `total_steps=40000 stop_at=400 prefix=speed` and read `steps_per_s`. Delete the speed runs afterwards.
- Long jobs: run them in the background, or under `nohup`, `tmux` or the cluster's scheduler, so they survive a disconnect.

## Code map
See the "Code" section of the README. The most relevant files for the next steps:
- `scripts/train.py`: training loop, validation by depth bucket, checkpoints.
- `rubik/pool.py`: pool loading, validation split, `CurriculumSampler`.
- `rubik/evaluate.py`: greedy and beam rollouts, progress measures, baseline policies, `model_policy`.
- `rubik/testsets.py`: `load_set`, `SET_NAMES`.
- `scripts/run_baselines.py`: how baselines were evaluated, as a template for the final evaluation script.
