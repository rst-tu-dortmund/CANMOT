"""CANMOT package and pinned in-tree nuScenes-devkit bootstrap."""

from pathlib import Path
import sys


_DEVKIT = Path(__file__).resolve().parents[2] / "dependencies_repos" / "nuscenes-devkit" / "python-sdk"
if _DEVKIT.is_dir() and str(_DEVKIT) not in sys.path:
    sys.path.insert(0, str(_DEVKIT))
