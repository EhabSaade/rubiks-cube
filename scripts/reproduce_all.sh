#!/usr/bin/env bash
# Rerun the whole study from scratch on a Linux machine with NVIDIA GPUs:
#
#     bash scripts/cloud_setup.sh
#     export RUBIK_STORAGE=$HOME/rubiks-cube; source $RUBIK_STORAGE/.venv/bin/activate
#     GPUS="0 1 2 3" RUBIK_WORKERS=20 bash scripts/reproduce_all.sh
#
# Steps: data (stops if the frozen test sets differ from the recorded checksums),
# baselines, 18 main runs (3 arms x seeds 0-5, 20 stages), 12 two-stage curriculum
# runs, evaluation (step-100,000 and best-validation checkpoints), beam-width sweep,
# clean timing (seeds 0-2) and the summary tables. Every step skips or resumes
# finished work, so the script can simply be started again. Results are written to
# results/ (compare with git diff); runs and checkpoints to $RUBIK_STORAGE.
# About 10 hours on 4 H200 GPUs and 20 CPU cores, most of it training and beam search.
set -euo pipefail
: "${RUBIK_STORAGE:?set RUBIK_STORAGE to the storage folder}"
cd "$(dirname "$0")/.."
GPUS=(${GPUS:-0})
R=$RUBIK_STORAGE/runs
mkdir -p "$R"

# run a list of commands, one process per GPU at a time, spread round-robin
run_queue() {
    local -a cmds=("$@")
    for i in "${!GPUS[@]}"; do
        (
            for j in $(seq "$i" "${#GPUS[@]}" $((${#cmds[@]} - 1))); do
                CUDA_VISIBLE_DEVICES=${GPUS[$i]} bash -c "${cmds[$j]}" || echo "FAILED: ${cmds[$j]}"
            done
        ) &
    done
    wait
}

echo "== data"
bash scripts/build_all.sh

echo "== baselines (written to $R/baselines.json; recorded copy: results/baselines.json)"
[ -f "$R/baselines.json" ] || CUDA_VISIBLE_DEVICES=${GPUS[0]} python scripts/run_baselines.py

echo "== training: 2 runs per GPU at a time"
SLOTS="${GPUS[*]} ${GPUS[*]}"
GPUS="$SLOTS" SEEDS="0 1 2 3 4 5" CONFIG=configs/wide.yaml \
    bash scripts/run_arms.sh prefix=final total_steps=100000 n_stages=20
GPUS="$SLOTS" SEEDS="0 1 2 3 4 5" ARMS="true_distance kociemba" CONFIG=configs/wide.yaml \
    bash scripts/run_arms.sh prefix=stages2 total_steps=100000 n_stages=2

MAIN=() S2=()
for s in 0 1 2 3 4 5; do
    for a in true_distance kociemba mixed; do MAIN+=("final_${a}_s$s"); done
    for a in true_distance kociemba; do S2+=("stages2_${a}_s$s"); done
done

echo "== evaluation (beam 4096)"
cmds=()
for r in "${MAIN[@]}"; do
    cmds+=("python scripts/evaluate_runs.py --runs $r && python scripts/evaluate_runs.py --checkpoint best_val --runs $r")
done
for r in "${S2[@]}"; do
    cmds+=("python scripts/evaluate_runs.py --runs $r --out_dir results/stages2_eval && python scripts/evaluate_runs.py --checkpoint best_val --runs $r --out_dir results/stages2_eval")
done
run_queue "${cmds[@]}"

echo "== beam-width sweep (widths 2 to 1024)"
cmds=()
for r in "${MAIN[@]}" "${S2[@]}"; do
    cmds+=("python scripts/beam_sweep.py --runs $r && python scripts/beam_sweep.py --checkpoint best_val --runs $r")
done
run_queue "${cmds[@]}"

echo "== clean timing (seeds 0-2, one process per GPU)"
cmds=()
for s in 0 1 2; do
    for a in true_distance kociemba mixed; do
        cmds+=("python scripts/time_eval.py --runs final_${a}_s$s && python scripts/time_eval.py --checkpoint best_val --runs final_${a}_s$s")
    done
done
run_queue "${cmds[@]}"

echo "== summary"
python scripts/summarize_eval.py
echo "reproduction finished"
