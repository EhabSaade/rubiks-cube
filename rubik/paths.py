"""Locations of large local files, kept off OneDrive.

Code lives in the repo; data, checkpoints and runs live under STORAGE_ROOT
(default D:\\rubiks-cube). Override with the RUBIK_STORAGE environment variable.
"""
import os
from pathlib import Path

STORAGE_ROOT = Path(os.environ.get("RUBIK_STORAGE", r"D:\rubiks-cube"))
DATA_DIR = STORAGE_ROOT / "data"
CHECKPOINT_DIR = STORAGE_ROOT / "checkpoints"
RUNS_DIR = STORAGE_ROOT / "runs"
