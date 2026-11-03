"""Compare freshly built data with the checksums recorded in results/.

    python scripts/check_data.py

Checks the distance table keys, the depth 5 and 6 hold-out and every frozen
test set. A match means the rebuilt data is identical to the original, so the
recorded baselines and pilot results still apply. Exits with status 1 on any
mismatch.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubik.paths import DATA_DIR  # noqa: E402
from rubik.testsets import TESTSET_DIR  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    checks = [("table keys", RESULTS / "table_d6.json", DATA_DIR / "table_d6.json", "keys_sha256"),
              ("hold-out", RESULTS / "holdout_d5_d6.json", DATA_DIR / "holdout_d5_d6.json", "states_sha256")]
    ok = True
    for name, ref, new, key in checks:
        a, b = json.loads(ref.read_text())[key], json.loads(new.read_text())[key]
        print(f"{name}: {'match' if a == b else 'MISMATCH'}")
        ok &= a == b
    ref = json.loads((RESULTS / "testsets_manifest.json").read_text())
    new = json.loads((TESTSET_DIR / "manifest.json").read_text())
    for name, entry in ref.items():
        if "states_sha256" not in entry:
            continue
        same = new.get(name, {}).get("states_sha256") == entry["states_sha256"]
        print(f"test set {name}: {'match' if same else 'MISMATCH'}")
        ok &= same
    print("all data matches the recorded checksums" if ok else "DATA DIFFERS from the recorded checksums")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
