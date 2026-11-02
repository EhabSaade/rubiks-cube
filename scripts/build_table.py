"""Build the exact distance table and save it under DATA_DIR.

    python scripts/build_table.py --depth 6
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubik.paths import DATA_DIR  # noqa: E402
from rubik.table import KNOWN_COUNTS, DistanceTable  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    t0 = time.time()
    table = DistanceTable.build(args.depth, args.device)
    counts = torch.bincount(table.dist.long()).tolist()
    ok = counts == KNOWN_COUNTS[:args.depth + 1]
    print("counts per depth:", counts, "match known:", ok)
    if not ok:
        sys.exit("count mismatch, table not saved")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"table_d{args.depth}.pt"
    table.save(path)
    sha = hashlib.sha256(table.keys.cpu().numpy().tobytes()).hexdigest()
    meta = {"depth": args.depth, "counts": counts, "keys_sha256": sha,
            "build_seconds": round(time.time() - t0, 1), "file": str(path)}
    (DATA_DIR / f"table_d{args.depth}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
