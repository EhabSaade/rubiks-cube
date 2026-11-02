import subprocess
import sys

from rubik import cube, koc_worker


def test_worker_move_names_match_simulator():
    assert koc_worker.MOVE_NAMES == cube.MOVE_NAMES


def test_worker_does_not_import_torch():
    code = "import sys, rubik.koc_worker; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
