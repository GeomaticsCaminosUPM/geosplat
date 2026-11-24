"""Tests for the targets workflow: folder resolution, GCP files of the targets folder, picks and the HTML picker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from geosplat.gcp import load_picks, refine_georeference
from geosplat.geo import GeoConfig, GeoReference, find_gcp_files
from geosplat.picker_html import save_picker_html
from geosplat.sources import is_within, resolve_subfolder

CRS = "EPSG:25830"
ORIGIN = np.array([400_000.0, 4_500_000.0, 800.0])
GCPS = {"A": np.array([400_010.0, 4_500_010.0, 830.0]), "B": np.array([400_030.0, 4_500_012.0, 831.0]),
        "C": np.array([400_020.0, 4_500_035.0, 829.0]), "D": np.array([400_005.0, 4_500_030.0, 830.5])}


def _geo() -> GeoReference:
    """A georeference whose scene frame is the local frame (identity).

    Returns:
        The georeference.
    """
    return GeoReference(CRS, ORIGIN.copy(), np.eye(4), 1.0, 10, {})


def _rasters(folder: Path) -> tuple[Path, Path]:
    """Write a small RGBA orthophoto and a flat DSM (z = 822 m) covering the GCPs.

    Args:
        folder: Output folder.

    Returns:
        Tuple ``(ortho_tif, dsm_tif)``.
    """
    tr = from_origin(399_990.0, 4_500_050.0, 0.5, 0.5)
    ortho, dsm = folder / "ortho.tif", folder / "dsm.tif"
    rgba = np.full((4, 120, 120), 200, np.uint8)
    with rasterio.open(ortho, "w", driver="GTiff", width=120, height=120, count=4, dtype="uint8", crs=CRS,
                       transform=tr) as dst:
        dst.write(rgba)
    with rasterio.open(dsm, "w", driver="GTiff", width=120, height=120, count=1, dtype="float32", crs=CRS,
                       transform=tr, nodata=-9999.0) as dst:
        dst.write(np.full((1, 120, 120), 822.0, np.float32))
    return ortho, dsm


def test_resolve_subfolder_prefers_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Relative folders are looked up in the dataset first, not the working directory."""
    (tmp_path / "data" / "flight b").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert resolve_subfolder(tmp_path / "data", "flight b") == (tmp_path / "data" / "flight b").resolve()
    assert resolve_subfolder(tmp_path / "data", ".") == (tmp_path / "data").resolve()
    assert is_within(tmp_path / "data" / "flight b" / "x.jpg", tmp_path / "data")
    with pytest.raises(FileNotFoundError):
        resolve_subfolder(tmp_path / "data", "missing")


def test_targets_folder_gcp_files_only(tmp_path: Path) -> None:
    """With targets_orto, only the GCP files of that folder are used."""
    from geosplat.geo import gcp_setup

    rows = "\n".join(f"{n},{x},{y},{z}" for n, (x, y, z) in GCPS.items())
    for sub, name in (("a", "targets_a.csv"), ("b", "targets_b.csv")):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / name).write_text(rows + "\n")
    assert len(find_gcp_files(tmp_path)) == 2
    cfg = GeoConfig(targets_orto="b", gcps_crs=CRS)
    path, crs = gcp_setup(tmp_path, cfg)
    assert Path(path).name == "targets_b.csv" and crs == CRS


def test_refine_with_page_picks_heights() -> None:
    """Picks with heights need no DSM and recover a shift exactly."""
    geo = _geo()
    shift = np.array([1.0, -0.5, -7.0])               # the current georeference is off by this
    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    picks = {}
    for n, (x, y, z) in GCPS.items():
        lon, lat = to_ll.transform(x + shift[0], y + shift[1])
        picks[n] = (lat, lon, z + shift[2])
    new, report = refine_georeference(geo, picks, GCPS, None)
    assert report["rmse_xy_m"] < 1e-3 and report["rmse_z_m"] < 1e-3
    np.testing.assert_allclose(new.scene_to_geo[:3, 3], -shift, atol=1e-3)


def test_picker_html_and_saved_picks(tmp_path: Path) -> None:
    """The page embeds tiles, GCPs, the DSM and the georeference; saved picks load in both formats."""
    ortho, dsm = _rasters(tmp_path)
    html = save_picker_html(ortho, dsm, GCPS, _geo(), tmp_path / "gcp" / "picker.html",
                            picks={"A": [40.6, -4.0, 830.0]}, scene_transform=np.eye(4))
    text = html.read_text()
    data = json.loads(text.split("const D = ")[1].split(";\nconst KEY")[0])
    assert {g["name"] for g in data["gcps"]} == set(GCPS)
    assert data["tiles"] and all((html.parent / t["url"]).is_file() for t in data["tiles"])
    assert data["dsm"]["fill"] == pytest.approx(822.0)
    assert data["geo"]["crs"] == CRS and data["picks"]["A"][2] == 830.0
    # Page format and notebook format.
    page = tmp_path / "page.json"
    page.write_text(json.dumps({"picks": {"A": [40.6, -4.0, 830.0]}, "geo": data["geo"]}))
    nb = tmp_path / "nb.json"
    nb.write_text(json.dumps({"A": [40.6, -4.0]}))
    assert load_picks(page) == {"A": (40.6, -4.0, 830.0)}
    assert load_picks(nb) == {"A": (40.6, -4.0)}
