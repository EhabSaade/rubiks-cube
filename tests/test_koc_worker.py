import subprocess
import sys

import torch

from rubik import cube, koc_worker
from rubik.cube import apply_sequence, is_solved, random_states
from rubik.kociemba_solver import solve_states


def test_worker_move_names_match_simulator():
    assert koc_worker.MOVE_NAMES == cube.MOVE_NAMES


def test_worker_does_not_import_torch():
    code = "import sys, rubik.koc_worker; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_solution_encoding_roundtrip():
    for sol in (None, [], [0, 17, 5]):
        assert koc_worker.decode_solution(koc_worker.encode_solution(sol)) == sol


def test_parallel_matches_serial_and_solves():
    s = random_states(40, torch.Generator().manual_seed(7))
    s[3] = cube.solved()[0]
    serial = solve_states(s, workers=1)
    parallel = solve_states(s, workers=3)
    assert serial == parallel
    assert parallel[3] == []
    for row, sol in zip(s, parallel):
        assert is_solved(apply_sequence(row.unsqueeze(0), sol)).all()
