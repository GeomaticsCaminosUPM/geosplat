"""Tests for photo sources: zip extraction and source resolution."""

from __future__ import annotations

import zipfile
from pathlib import Path

from geosplat.datasets import resolve_image_source, unzip_images


def _make_zip(path: Path) -> Path:
    """Create a zip with two flight folders, a CSV, a video, macOS junk and a non-photo file.

    Args:
        path: Zip path to create.

    Returns:
        The zip path.
    """
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("flight/a/IMG_1.JPG", b"x")
        zf.writestr("flight/b/IMG_2.jpg", b"x")
        zf.writestr("flight/b/IMG_1.JPG", b"y")          # same name in another subfolder
        zf.writestr("flight/targets.csv", b"name,x,y,z\nA,1,2,3\n")
        zf.writestr("flight/b/clip.mp4", b"v")
        zf.writestr("__MACOSX/flight/._IMG_1.JPG", b"junk")
        zf.writestr("flight/.DS_Store", b"junk")
        zf.writestr("flight/notes.docx", b"junk")
    return path


def test_unzip_keeps_folders_and_filters(tmp_path: Path) -> None:
    """Folders are kept (wrapping folder dropped); photos, videos and CSVs extracted; junk skipped.

    Args:
        tmp_path: Pytest temporary directory.
    """
    out = unzip_images(_make_zip(tmp_path / "flight1.zip"))
    assert out == tmp_path / "flight1"
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert files == ["a/IMG_1.JPG", "b/IMG_1.JPG", "b/IMG_2.jpg", "b/clip.mp4", "targets.csv"]


def test_resolve_sources(tmp_path: Path) -> None:
    """Zips, dataset folders with a zip, and missing folders with a sibling zip resolve to photo folders.

    Args:
        tmp_path: Pytest temporary directory.
    """
    assert resolve_image_source(_make_zip(tmp_path / "images.zip")) == tmp_path / "images"
    ds = tmp_path / "dataset"
    ds.mkdir()
    _make_zip(ds / "photos.zip")
    assert resolve_image_source(ds) == ds / "images"
    assert (ds / "images" / "b" / "IMG_2.jpg").is_file()
    _make_zip(tmp_path / "flight.zip")
    assert resolve_image_source(tmp_path / "flight") == tmp_path / "flight"
    assert resolve_image_source(ds) == ds / "images"                 # already extracted: reused


def test_gcp_csv_header_optional(tmp_path: Path) -> None:
    """GCP files load the same with or without a header, with any common delimiter.

    Args:
        tmp_path: Pytest temporary directory.
    """
    from geosplat.geo import find_gcp_file, read_gcps

    rows = "D1,411339.128,4496352.532,839.571\nD2,411345.226,4496318.775,837.526\nD3,411370.576,4496335.837,837.214\n"
    variants = {
        "no_header.csv": rows,
        "header.csv": "name,x,y,z\n" + rows,
        "semicolon.csv": "id;E;N;H\n" + rows.replace(",", ";"),
        "spaces.txt": rows.replace(",", " "),
    }
    expected = read_gcps(_write(tmp_path / "ref.csv", rows))
    assert list(expected) == ["D1", "D2", "D3"]
    for name, text in variants.items():
        got = read_gcps(_write(tmp_path / name, text))
        assert got.keys() == expected.keys(), name
        assert all((got[k] == expected[k]).all() for k in got), name
    folder = tmp_path / "photos"
    folder.mkdir()
    _write(folder / "targets.csv", rows)
    assert find_gcp_file(folder) == folder / "targets.csv"


def _write(path: Path, text: str) -> Path:
    """Write a text file.

    Args:
        path: File path.
        text: Content.

    Returns:
        The path.
    """
    path.write_text(text)
    return path
