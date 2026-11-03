#!/usr/bin/env bash
# Build all data from scratch on this machine, then check it against the
# recorded checksums. Every step is resumable or skips work already done, so
# the script can simply be started again after an interruption.
#
#     RUBIK_WORKERS=32 bash scripts/build_all.sh
#
# RUBIK_WORKERS is the number of Kociemba subprocesses (about the number of CPU
# cores; default 6). Needs RUBIK_STORAGE, a GPU and the kociemba package.
# Laptop times (RTX 3060, 6 workers): table 15 s, test sets about 2 hours,
# pool about 6 hours; Kociemba parts scale with workers, exact search with the GPU.
set -euo pipefail
: "${RUBIK_STORAGE:?set RUBIK_STORAGE to the storage folder}"
cd "$(dirname "$0")/.."
python -c "import kociemba" || { echo "kociemba is required to build data"; exit 1; }

D="$RUBIK_STORAGE/data"
[ -f "$D/table_d6.pt" ] || python scripts/build_table.py --depth 6
[ -f "$D/holdout_d5_d6.pt" ] || python scripts/make_holdout.py
python scripts/build_testsets.py
python scripts/check_data.py
python scripts/build_pool.py --part path --target 20000000
python scripts/build_pool.py --part exact
python scripts/build_pool.py --part shallow
echo "data build finished"
