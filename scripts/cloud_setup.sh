#!/usr/bin/env bash
# Set up a Linux GPU machine (e.g. an A100 instance) for training.
#
# Locally, pack the data once (Git Bash, from the repo folder):
#     bash scripts/pack_data.sh            # writes D:/rubiks-cube/rubik_data.tar + .sha256
# Upload rubik_data.tar and rubik_data.tar.sha256 to the machine, clone the repo, then:
#     bash scripts/cloud_setup.sh /path/to/rubik_data.tar
#     export RUBIK_STORAGE=$HOME/rubiks-cube
#     python scripts/train.py --config configs/base.yaml --arm true_distance --seed 0
set -euo pipefail

TAR=${1:?usage: cloud_setup.sh /path/to/rubik_data.tar}
STORAGE=${RUBIK_STORAGE:-$HOME/rubiks-cube}
REPO=$(cd "$(dirname "$0")/.." && pwd)

(cd "$(dirname "$TAR")" && sha256sum -c "$(basename "$TAR").sha256")
mkdir -p "$STORAGE"
tar -xf "$TAR" -C "$STORAGE"

python3 -m venv "$STORAGE/.venv"
source "$STORAGE/.venv/bin/activate"
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r "$REPO/requirements.txt"

cd "$REPO"
RUBIK_STORAGE="$STORAGE" python - <<'EOF'
import torch
from rubik.pool import Pool
assert torch.cuda.is_available(), "no GPU visible"
print("GPU:", torch.cuda.get_device_name(0))
print("pool:", f"{len(Pool.load()):,}", "cubes")
EOF
RUBIK_STORAGE="$STORAGE" python -m pytest -q tests
echo "ready: export RUBIK_STORAGE=$STORAGE; source $STORAGE/.venv/bin/activate"
