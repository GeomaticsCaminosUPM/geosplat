"""Tests for photo positions from external sources (geo.txt, CSV, sidecars, vector, MRK, GPX)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pyproj import Transformer

from geosplat.geo import find_gcp_file, read_gps
from geosplat.positions import photo_positions

ROOT = Path(__file__).resolve().parents[1]
LONLAT = [(-4.0485, 40.6128), (-4.0480, 40.6129), (-4.0476, 40.6131), (-4.0478, 40.6135), (-4.0483, 40.6136)]
ALT = 857.0


def _photos(folder: Path, times: list[str] | None = None) -> list[Path]:
    """Write tiny JPEGs without GPS (optionally with DateTimeOriginal).

    Args:
        folder: Destination folder.
        times: EXIF capture times ``YYYY:MM:DD HH:MM:SS`` per photo.

    Returns:
        Photo paths ``img_1.jpg`` ... ``img_5.jpg``.
    """
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(len(LONLAT)):
        p = folder / f"img_{i + 1}.jpg"
        exif = Image.Exif()
        if times:
            exif.get_ifd(0x8769)[0x9003] = times[i]      # DateTimeOriginal
        Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(p, exif=exif.tobytes())
        paths.append(p)
    return paths


def _check(pos: dict, source: str | None = None, atol: float = 1e-6) -> None:
    """Assert positions match :data:`LONLAT` for all photos.

    Args:
        pos: Result of ``photo_positions``.
        source: Expected ``source`` attribute (``None`` skips the check).
        atol: Tolerance in degrees.
    """
    assert len(pos) == len(LONLAT), pos.keys()
    for i, (lon, lat) in enumerate(LONLAT):
        p = pos[f"img_{i + 1}"]
        assert abs(p.lon - lon) < atol and abs(p.lat - lat) < atol, (i, p)
        if source:
            assert p.source == source


def test_odm_geo_txt_in_parent(tmp_path: Path) -> None:
    """ODM geo.txt (CRS header, projected coordinates) next to images/ is used.

    Args:
        tmp_path: Pytest temporary directory.
    """
    _photos(tmp_path / "images")
    t = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)
    xs, ys = t.transform(*zip(*LONLAT))
    lines = ["EPSG:25830"] + [f"img_{i + 1}.jpg {x:.3f} {y:.3f} {ALT}" for i, (x, y) in enumerate(zip(xs, ys))]
    (tmp_path / "geo.txt").write_text("\n".join(lines) + "\n")
    pos = photo_positions(tmp_path / "images")
    _check(pos, "geo.txt", atol=1e-7)
    assert pos["img_1"].alt == ALT


def test_csv_variants(tmp_path: Path) -> None:
    """Metashape-style CSV with header, headerless CSV and a projected table with a given CRS.

    Args:
        tmp_path: Pytest temporary directory.
    """
    a = tmp_path / "a"
    _photos(a)
    (a / "cameras.csv").write_text("#Label,Longitude,Latitude,Altitude,Yaw,Pitch,Roll\n" + "".join(
        f"img_{i + 1}.JPG,{lon},{lat},{ALT},90,-30,0\n" for i, (lon, lat) in enumerate(LONLAT)))
    pos = photo_positions(a)
    _check(pos, "cameras.csv")
    assert pos["img_2"].yaw == 90 and pos["img_2"].pitch == -30
    b = tmp_path / "b"
    _photos(b)
    (b / "pos.txt").write_text("".join(f"img_{i + 1} {lon} {lat} {ALT}\n" for i, (lon, lat) in enumerate(LONLAT)))
    _check(photo_positions(b), "pos.txt")
    c = tmp_path / "c"
    _photos(c)
    xs, ys = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True).transform(*zip(*LONLAT))
    (c / "utm.csv").write_text("name;x;y;z\n" + "".join(f"img_{i + 1};{x};{y};{ALT}\n" for i, (x, y) in enumerate(zip(xs, ys))))
    assert photo_positions(c) == {}                                   # projected, no CRS: not guessed
    _check(photo_positions(c, crs="EPSG:32630"), "utm.csv")


def test_xmp_sidecars_and_vector_and_mrk(tmp_path: Path) -> None:
    """XMP sidecars, a GeoJSON with a filename field and a DJI MRK file.

    Args:
        tmp_path: Pytest temporary directory.
    """
    a = tmp_path / "xmp"
    for p, (lon, lat) in zip(_photos(a), LONLAT):
        deg, mins = int(abs(lat)), (abs(lat) - int(abs(lat))) * 60
        (a / f"{p.stem}.xmp").write_text(
            f'<x:xmpmeta><rdf:Description exif:GPSLatitude="{deg},{mins:.6f}N" exif:GPSLongitude="{lon}" '
            f'exif:GPSAltitude="857000/1000" drone-dji:GimbalYawDegree="+155.3" drone-dji:GimbalPitchDegree="-90.0"/></x:xmpmeta>')
    pos = photo_positions(a)
    _check(pos, atol=1e-5)
    assert pos["img_1"].pitch == -90.0 and abs(pos["img_1"].alt - 857.0) < 1e-6

    pyogrio = pytest.importorskip("pyogrio")
    import shapely
    b = tmp_path / "vec"
    _photos(b)
    geoms = shapely.to_wkb([shapely.Point(lon, lat, ALT) for lon, lat in LONLAT])
    pyogrio.raw.write(str(b / "cams.geojson"), np.array(geoms, dtype=object),
                      [np.array([f"img_{i + 1}.jpg" for i in range(len(LONLAT))], dtype=object)], ["filename"],
                      geometry_type="Point Z", crs="EPSG:4326", driver="GeoJSON")
    _check(photo_positions(b), "cams.geojson")

    c = tmp_path / "mrk"
    _photos(c)
    (c / "DJI_202411_Timestamp.MRK").write_text("".join(
        f"{i + 1}\t394003.39\t[2112]\t  -12,N\t  8,E\t -183,V\t{lat:.8f},Lat\t{lon:.8f},Lon\t{ALT},Ellh\t0.01, 0.01, 0.02\t50,Q\n"
        for i, (lon, lat) in enumerate(LONLAT)))
    _check(photo_positions(c), "DJI_202411_Timestamp.MRK", atol=1e-7)


def test_gpx_with_camera_clock_offset(tmp_path: Path) -> None:
    """Photos are placed on a GPX track by time; a +1 h camera clock offset is estimated.

    Args:
        tmp_path: Pytest temporary directory.
    """
    times = [f"2024:04:11 19:28:{10 * i:02d}" for i in range(len(LONLAT))]   # local time = UTC+1
    _photos(tmp_path, times)
    pts = "".join(f'<trkpt lat="{lat}" lon="{lon}"><ele>{ALT}</ele><time>2024-04-11T18:28:{10 * i:02d}Z</time></trkpt>'
                  for i, (lon, lat) in enumerate(LONLAT))
    (tmp_path / "track.gpx").write_text(f"<gpx><trk><trkseg>{pts}</trkseg></trk></gpx>")
    _check(photo_positions(tmp_path), "track.gpx")


def test_positions_csv_is_not_a_gcp_file(tmp_path: Path) -> None:
    """A CSV listing the photos is used as positions and never mistaken for ground control points.

    Args:
        tmp_path: Pytest temporary directory.
    """
    _photos(tmp_path)
    (tmp_path / "positions.csv").write_text("".join(f"img_{i + 1},{lon},{lat},{ALT}\n" for i, (lon, lat) in enumerate(LONLAT)))
    (tmp_path / "targets.csv").write_text("D1,-4.0482,40.6131,839\nD2,-4.0479,40.6133,837\nD3,-4.0478,40.6130,836\n")
    assert find_gcp_file(tmp_path) == tmp_path / "targets.csv"
    _check(photo_positions(tmp_path), "positions.csv")


def test_exif_positions_still_used() -> None:
    """Photos with GPS EXIF (drone2024) keep their EXIF positions."""
    images = ROOT / "examples" / "drone2024" / "images"
    if not images.is_dir():
        pytest.skip("drone2024 photos not present")
    pos = photo_positions(images)
    first = sorted(images.glob("*.JPG"))[0]
    lat, lon, _ = read_gps(first)
    p = pos[first.stem.lower()]
    assert p.source == "exif" and abs(p.lat - lat) < 1e-9 and abs(p.lon - lon) < 1e-9 and p.pitch == -90.0
