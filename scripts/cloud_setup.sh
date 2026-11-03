#!/usr/bin/env bash
# Set up a Linux GPU machine. Needs Python 3.10 or newer.
#
# Fresh data (built on this machine afterwards with scripts/build_all.sh;
# needs gcc and Python headers for kociemba):
#     bash scripts/cloud_setup.sh
# Or with data copied from another machine (made there with scripts/pack_data.sh):
#     bash scripts/cloud_setup.sh /path/to/rubik_data.tar
#     export RUBIK_STORAGE=$HOME/rubiks-cube; source $RUBIK_STORAGE/.venv/bin/activate
#     python scripts/train.py --config configs/base.yaml --arm true_distance --seed 0
set -euo pipefail

TAR=${1:-}
STORAGE=${RUBIK_STORAGE:-$HOME/rubiks-cube}
PYTHON=${PYTHON:-python3}
REPO=$(cd "$(dirname "$0")/.." && pwd)

"$PYTHON" -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ needed'"
mkdir -p "$STORAGE"
if [ -n "$TAR" ]; then
    (cd "$(dirname "$TAR")" && sha256sum -c "$(basename "$TAR").sha256")
    tar -xf "$TAR" -C "$STORAGE"
fi

"$PYTHON" -m venv "$STORAGE/.venv"
source "$STORAGE/.venv/bin/activate"
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
grep -v '^kociemba' "$REPO/requirements.txt" | pip install -r /dev/stdin
# kociemba is needed to build data, not to train or evaluate models
if [ -n "$TAR" ]; then
    pip install kociemba || echo "WARNING: kociemba did not install (needs gcc and Python headers); training still works"
else
    pip install kociemba || { echo "ERROR: kociemba is needed to build data; install gcc and python3-dev, then rerun"; exit 1; }
fi

cd "$REPO"
RUBIK_STORAGE="$STORAGE" python - <<'EOF'
import torch
from rubik.pool import Pool
assert torch.cuda.is_available(), "no GPU visible"
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:", torch.cuda.get_device_name(i))
try:
    print("pool:", f"{len(Pool.load()):,}", "cubes")
except Exception as e:
    print("no pool yet (build it with scripts/build_all.sh):", type(e).__name__)
EOF
RUBIK_STORAGE="$STORAGE" python -m pytest -q tests || echo "WARNING: some tests failed (test_koc_worker needs kociemba)"
echo "ready: export RUBIK_STORAGE=$STORAGE; source $STORAGE/.venv/bin/activate"
