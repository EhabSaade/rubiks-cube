"""Reserve the depth 5 and 6 hold-out from the saved table.

    python scripts/make_holdout.py
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubik.holdout import HoldoutFilter, make_holdout  # noqa: E402
from rubik.paths import DATA_DIR  # noqa: E402
from rubik.table import DistanceTable  # noqa: E402


def main():
    path = DATA_DIR / "holdout_d5_d6.pt"
    if path.exists():
        sys.exit(f"{path} already exists; the hold-out is frozen. Delete it only on purpose.")
    table = DistanceTable.load(DATA_DIR / "table_d6.pt")
    h = make_holdout(table)
    assert HoldoutFilter(h["states"]).contains(h["states"]).all()
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in h.items()}, path)
    counts = {int(d): int((h["dist"] == d).sum()) for d in h["depths"]}
    sha = hashlib.sha256(h["states"].cpu().numpy().tobytes()).hexdigest()
    meta = {"file": str(path), "share": h["share"], "seed": h["seed"], "counts": counts,
            "states_sha256": sha}
    (DATA_DIR / "holdout_d5_d6.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
