#!/usr/bin/env bash
# Set up a Linux GPU machine for training. Needs Python 3.10 or newer.
#
# Locally, pack the data once (Git Bash, from the repo folder):
#     bash scripts/pack_data.sh            # writes D:/rubiks-cube/rubik_data.tar + .sha256
# Copy rubik_data.tar and rubik_data.tar.sha256 to the machine, clone the repo, then:
#     bash scripts/cloud_setup.sh /path/to/rubik_data.tar
#     export RUBIK_STORAGE=$HOME/rubiks-cube; source $RUBIK_STORAGE/.venv/bin/activate
#     python scripts/train.py --config configs/base.yaml --arm true_distance --seed 0
set -euo pipefail

TAR=${1:?usage: cloud_setup.sh /path/to/rubik_data.tar}
STORAGE=${RUBIK_STORAGE:-$HOME/rubiks-cube}
PYTHON=${PYTHON:-python3}
REPO=$(cd "$(dirname "$0")/.." && pwd)

"$PYTHON" -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ needed'"
(cd "$(dirname "$TAR")" && sha256sum -c "$(basename "$TAR").sha256")
mkdir -p "$STORAGE"
tar -xf "$TAR" -C "$STORAGE"

"$PYTHON" -m venv "$STORAGE/.venv"
source "$STORAGE/.venv/bin/activate"
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
grep -v '^kociemba' "$REPO/requirements.txt" | pip install -r /dev/stdin
# kociemba is only needed to build data, not to train or evaluate models
pip install kociemba || echo "WARNING: kociemba did not install (needs gcc and Python headers); training still works"

cd "$REPO"
RUBIK_STORAGE="$STORAGE" python - <<'EOF'
import torch
from rubik.pool import Pool
assert torch.cuda.is_available(), "no GPU visible"
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:", torch.cuda.get_device_name(i))
print("pool:", f"{len(Pool.load()):,}", "cubes")
EOF
RUBIK_STORAGE="$STORAGE" python -m pytest -q tests || echo "WARNING: some tests failed (test_koc_worker needs kociemba)"
echo "ready: export RUBIK_STORAGE=$STORAGE; source $STORAGE/.venv/bin/activate"
