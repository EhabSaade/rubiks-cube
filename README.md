# Rubik's Cube Move Prediction

Can a neural network predict strong or optimal next moves on a 3x3 cube, and does curriculum learning help?

## Setup (Windows)
```
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install -r requirements.txt
```

## Tests
```
.venv\Scripts\python -m pytest -q tests
```

## Layout
- `rubik/cube.py`: batched cubie simulator (18 half-turn moves, Kociemba conventions)
- `rubik/facelet.py`: conversion to and from Kociemba facelet strings
- `rubik/encoding.py`: 480-value cubie one-hot encoding
- `rubik/scramble.py`: non-redundant scramble generator
- `data/`, `checkpoints/`, `runs/`: local only, not in git
