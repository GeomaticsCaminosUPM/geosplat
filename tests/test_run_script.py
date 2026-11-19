"""End-to-end test of ``examples/run.py`` with the ``very_fast`` preset (``pytest --run-slow``)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["dron2024", "south_building"])
def test_run_script_very_fast(dataset: str) -> None:
    """``run.py`` produces every product for a dataset.

    Args:
        dataset: Folder name in ``examples/`` (skipped if its photos are missing).
    """
    if not (ROOT / "examples" / dataset / "images").is_dir():
        pytest.skip(f"examples/{dataset}/images not present")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "run.py"), dataset, "--preset", "very_fast", "--video-frames", "0"],
        cwd=ROOT, capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    out = ROOT / "examples" / dataset / "output_very_fast"
    for product in ("splat.ply", "viewer.html", "points_sparse.ply", "points_dense.ply",
                    "fast_orto/fast_orto.tif", "ortho/orthophoto.tif", "ortho/dsm.tif"):
        assert (out / product).is_file(), product
