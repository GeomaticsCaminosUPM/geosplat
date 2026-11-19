"""Tests for ground control point files: CRS from vector files, CRS inference for CSV, conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pyproj import Transformer

from geosplat.geo import GeoConfig, gcp_setup, infer_gcp_crs, load_gcps, read_gcp_file

ROOT = Path(__file__).resolve().parents[1]
# Photo positions of a small flight near Madrid (lon, lat) and GCPs inside it.
PHOTOS = np.array([[-4.0485, 40.6128], [-4.0476, 40.6128], [-4.0476, 40.6136], [-4.0485, 40.6136]])
GCP_LONLAT = {"A": (-4.0482, 40.6131, 839.5), "B": (-4.0479, 40.6133, 837.2), "C": (-4.0478, 40.6130, 836.0)}


def _gcps_in(crs: str) -> dict[str, np.ndarray]:
    """GCPs converted to a CRS.

    Args:
        crs: Target CRS.

    Returns:
        ``{name: [x, y, z]}``.
    """
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return {n: np.array([*t.transform(lon, lat), z]) for n, (lon, lat, z) in GCP_LONLAT.items()}


def test_infer_crs() -> None:
    """Projected (ETRS89 UTM 30N preferred in Europe) and geographic coordinates are recognised."""
    assert infer_gcp_crs(_gcps_in("EPSG:25830"), PHOTOS)[0] == "EPSG:25830"
    assert infer_gcp_crs(_gcps_in("EPSG:4326"), PHOTOS)[0] == "EPSG:4326"
    far = {n: v + np.array([50_000.0, 0, 0]) for n, v in _gcps_in("EPSG:25830").items()}
    assert infer_gcp_crs(far, PHOTOS)[0] is None


def test_vector_file_crs_and_conversion(tmp_path: Path) -> None:
    """A GeoPackage carries its CRS; points are converted to the output CRS.

    Args:
        tmp_path: Pytest temporary directory.
    """
    pyogrio = pytest.importorskip("pyogrio")
    import shapely

    pts = _gcps_in("EPSG:25830")
    geoms = shapely.to_wkb([shapely.Point(*v) for v in pts.values()])
    path = tmp_path / "targets.gpkg"
    pyogrio.raw.write(str(path), np.array(geoms, dtype=object), [np.array(list(pts), dtype=object)], ["name"],
                      geometry_type="Point Z", crs="EPSG:25830", driver="GPKG")
    read, crs = read_gcp_file(path)
    assert crs is not None and "25830" in crs
    assert np.allclose(read["B"], pts["B"])
    utm_wgs84 = load_gcps(path, "EPSG:32630")
    assert np.allclose(utm_wgs84["B"], _gcps_in("EPSG:32630")["B"], atol=0.01)


def test_gcp_setup_sets_output_crs(tmp_path: Path) -> None:
    """A projected GCP CRS (given or inferred) becomes the output CRS when none is configured.

    Args:
        tmp_path: Pytest temporary directory.
    """
    csv = tmp_path / "targets.csv"
    csv.write_text("".join(f"{n},{x},{y},{z}\n" for n, (x, y, z) in _gcps_in("EPSG:25830").items()))
    cfg = GeoConfig(gcps=csv, gcps_crs="EPSG:25830")
    assert gcp_setup(tmp_path, cfg) == (csv, "EPSG:25830") and cfg.crs == "EPSG:25830"
    cfg = GeoConfig(gcps=csv, crs="EPSG:32630", gcps_crs="EPSG:25830")
    gcp_setup(tmp_path, cfg)
    assert cfg.crs == "EPSG:32630"                     # an explicit output CRS is kept


def test_drone2024_targets_crs() -> None:
    """The real drone2024 targets (ETRS89 / UTM 30N, no header-defined CRS) are recognised."""
    images = ROOT / "examples" / "drone2024" / "images"
    if not (images / "targets.csv").is_file():
        pytest.skip("drone2024 photos not present")
    cfg = GeoConfig()
    path, crs = gcp_setup(images, cfg)
    assert path.name == "targets.csv" and crs == "EPSG:25830" and cfg.crs == "EPSG:25830"
