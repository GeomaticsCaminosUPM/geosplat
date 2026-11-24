"""End-to-end test of ``examples/run.py`` with the ``very_fast`` preset (``pytest --run-slow``)."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["drone2024", "south_building", "seneca", "aerial_gcps"])
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


@pytest.mark.slow
def test_picks_round_trip(tmp_path: Path) -> None:
    """Picks saved from ``targets/locate_targets.html`` correct the georeference on the next run, reusing SfM and the splat.

    Picks are simulated 1 m east of the surveyed targets, with heights taken from
    the page's embedded DSM; the second run must remove that shift.

    Args:
        tmp_path: Temporary folder (output).
    """
    images = ROOT / "examples" / "aerial_gcps" / "images"
    if not images.is_dir():
        pytest.skip("examples/aerial_gcps/images not present (run examples/aerial_gcps/download.py)")
    out = tmp_path / "out"
    cmd = [sys.executable, str(ROOT / "examples" / "run.py"), "aerial_gcps", "--preset", "very_fast",
           "--video-frames", "0", "-o", str(out)]
    first = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    page = (out / "targets" / "locate_targets.html").read_text()
    data = json.loads(page.split("const D = ")[1].split(";\nconst KEY")[0])
    dsm = data["dsm"]
    picks = {}
    for g in data["gcps"]:
        lon = g["lon"] + 1.0 / (111_320 * math.cos(math.radians(g["lat"])))
        x = lon * 20037508.342789244 / 180
        y = math.log(math.tan((90 + g["lat"]) * math.pi / 360)) * 6378137
        c, r = int((x - dsm["west"]) / dsm["dx"]), int((y - dsm["north"]) / dsm["dy"])
        z = dsm["z"][r * dsm["w"] + c] if 0 <= c < dsm["w"] and 0 <= r < dsm["h"] else None
        picks[g["name"]] = [g["lat"], lon, z if z is not None else dsm["fill"]]
    (out / "targets" / "located_targets.json").write_text(json.dumps({"picks": picks, "geo": data["geo"],
                                                         "scene_transform": data["scene_transform"]}))
    second = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    log = second.stdout + second.stderr
    assert second.returncode == 0, log[-3000:]
    assert "Georeference corrected with 3 picked targets" in log
    assert "Reusing the trained splat" in log
    report = json.loads((out / "targets" / "report.json").read_text())
    shifts = [v[0] for v in report["shift_before_m"].values()]
    assert all(abs(s + 1.0) < 0.05 for s in shifts), shifts      # the 1 m east offset is removed
    assert json.loads((out / "geo.json").read_text())["georeferencing"].startswith("GCPs")
    shutil.rmtree(out)
