# Rubik's Cube Move Prediction

Can a neural network predict strong or optimal next moves on a 3x3 cube, and does curriculum learning help?

A policy network (residual MLP) is trained to predict the next move of a 3x3 cube in the half-turn metric. Three training arms are compared: a curriculum ordered by true distance, a curriculum ordered by Kociemba solution length, and a mixed-depth control. Labels come from an exact distance table (all states up to depth 6, exact labels to depth 7), exact GPU search up to distance 13, and the Kociemba two-phase solver beyond.

Method decisions and measurements are logged in [docs/decisions.md](docs/decisions.md).

## Layout
- Code: this repository
- Large files (Python environment, data, checkpoints, runs): a separate storage folder, `D:\rubiks-cube` by default. Set the `RUBIK_STORAGE` environment variable to use another location (see `rubik/paths.py`).

## Setup (Windows)
Needs Python 3.10 and Microsoft C++ Build Tools ("Desktop development with C++") for `kociemba`.
```
py -3.10 -m venv D:\rubiks-cube\.venv
D:\rubiks-cube\.venv\Scripts\python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
D:\rubiks-cube\.venv\Scripts\python -m pip install -r requirements.txt
```

## Building the data
```
python scripts/build_table.py --depth 6        # exact distance table, about 15 s on a GPU
python scripts/make_holdout.py                 # frozen 5% hold-out of depths 5 and 6
python scripts/build_testsets.py               # frozen test sets, about 2 hours (depth 13 dominates)
python scripts/build_pool.py --part path       # training pool, Kociemba path labeling
python scripts/build_pool.py --part exact      # exact pools at distances 10 to 12
python scripts/build_pool.py --part shallow    # Kociemba lengths for depth 1 to 7 cubes
python scripts/run_baselines.py                # random, shallow search, Kociemba baselines
```

## Training
```
python scripts/train.py --config configs/base.yaml --arm true_distance --seed 0
```
Arms: `true_distance`, `kociemba`, `mixed`. Runs resume from their last checkpoint.

All arms and seeds on several GPUs (Linux; setup with `scripts/pack_data.sh` and `scripts/cloud_setup.sh`):
```
GPUS="0 1 2" bash scripts/run_arms.sh prefix=final
```

Project state and open decisions: [CONTEXT.md](CONTEXT.md). Instructions for AI agents: [AGENTS.md](AGENTS.md).

## Tests
```
python -m pytest -q tests
```

## Code
- `rubik/cube.py`: batched cubie simulator (18 half-turn moves, Kociemba conventions)
- `rubik/facelet.py`: conversion to and from Kociemba facelet strings
- `rubik/encoding.py`: 480-value cubie one-hot encoding
- `rubik/scramble.py`: non-redundant scramble generator
- `rubik/table.py`: exact distance table with all optimal moves
- `rubik/exact_search.py`, `rubik/batch_label.py`: exact labels beyond the table
- `rubik/kociemba_solver.py`, `rubik/koc_worker.py`: Kociemba wrapper with parallel subprocess workers
- `rubik/holdout.py`, `rubik/testsets.py`: frozen hold-out and test sets, training isolation filter
- `rubik/evaluate.py`: greedy and beam rollouts, progress measures, baseline policies
- `rubik/pool.py`: training pool and curriculum sampler
- `rubik/model.py`: residual MLP policy
- `rubik/paths.py`: storage locations
