#!/usr/bin/env bash
# Pack the data needed for training and evaluation into one tar with a checksum,
# for upload to a cloud GPU machine (see scripts/cloud_setup.sh).
set -euo pipefail

STORAGE=${RUBIK_STORAGE:-/d/rubiks-cube}
cd "$STORAGE"
tar -cf rubik_data.tar data/pool data/testsets data/table_d6.pt data/table_d6.json \
    data/holdout_d5_d6.pt data/holdout_d5_d6.json
sha256sum rubik_data.tar > rubik_data.tar.sha256
ls -l rubik_data.tar
cat rubik_data.tar.sha256
