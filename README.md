# Rubik's Cube Move Prediction

Can a neural network predict strong or optimal next moves on a 3x3 cube, and does curriculum learning help?

## Layout on this machine
- Code (this repo): `Desktop\AI project`, synced by OneDrive
- Environment, data, checkpoints, runs: `D:\rubiks-cube` (set `RUBIK_STORAGE` to change; see `rubik/paths.py`)

## Setup (Windows)
Needs Microsoft C++ Build Tools ("Desktop development with C++") for `kociemba`.
```
py -3.10 -m venv D:\rubiks-cube\.venv
D:\rubiks-cube\.venv\Scripts\python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
D:\rubiks-cube\.venv\Scripts\python -m pip install -r requirements.txt
```

## Tests
```
D:\rubiks-cube\.venv\Scripts\python -m pytest -q tests
```

## Code
- `rubik/cube.py`: batched cubie simulator (18 half-turn moves, Kociemba conventions)
- `rubik/facelet.py`: conversion to and from Kociemba facelet strings
- `rubik/encoding.py`: 480-value cubie one-hot encoding
- `rubik/scramble.py`: non-redundant scramble generator
- `rubik/paths.py`: where large local files are stored
