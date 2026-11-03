#!/usr/bin/env bash
# Run every (arm, seed) pair, spread over the given GPUs. Each GPU works through
# its share one run at a time; runs resume from their last checkpoint, so the
# script can simply be started again after an interruption.
#
#     GPUS="0 1 2" CONFIG=configs/wide.yaml bash scripts/run_arms.sh prefix=final total_steps=100000
#
# Arguments are passed to train.py as --set overrides. Environment:
#   GPUS    GPU ids to use (default "0")
#   CONFIG  config file (default configs/base.yaml)
#   SEEDS   seeds (default "0 1 2")
#   ARMS    arms (default "true_distance kociemba mixed")
set -uo pipefail

GPUS=(${GPUS:-0})
CONFIG=${CONFIG:-configs/base.yaml}
SEEDS=(${SEEDS:-0 1 2})
ARMS=(${ARMS:-true_distance kociemba mixed})
STORAGE=${RUBIK_STORAGE:?set RUBIK_STORAGE to the storage folder}
cd "$(dirname "$0")/.."
mkdir -p "$STORAGE/runs"

jobs=()
for s in "${SEEDS[@]}"; do for a in "${ARMS[@]}"; do jobs+=("$a $s"); done; done

for i in "${!GPUS[@]}"; do
    (
        for j in $(seq "$i" "${#GPUS[@]}" $((${#jobs[@]} - 1))); do
            read -r arm seed <<< "${jobs[$j]}"
            out="$STORAGE/runs/launch_${arm}_s${seed}.out"
            echo "GPU ${GPUS[$i]}: $arm seed $seed -> $out"
            CUDA_VISIBLE_DEVICES=${GPUS[$i]} python scripts/train.py --config "$CONFIG" \
                --arm "$arm" --seed "$seed" --set "$@" > "$out" 2>&1 \
                || echo "FAILED: $arm seed $seed (see $out)"
        done
    ) &
done
wait
echo "all runs finished"
