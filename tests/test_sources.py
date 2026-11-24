"""Tests for dataset scanning: subfolders, ignored folders, videos, SRT tracks, merged GCP files."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from geosplat.geo import find_gcp_files, read_gcp_files, read_gcps
from geosplat.sources import extract_frames, ffmpeg_executable, read_dji_srt, scan_dataset

TARGETS = "1,411341.113,4496351.354,949.579,D1\n2,411345.289,4496323.242,947.590,D2\n3,411368.998,4496333.474,947.142,D3\n"
TARGETS_SC = "1,411341.113,4496351.354,949.579\n2,411345.289,4496323.242,947.590\n3,411368.998,4496333.474,947.142\n"


def _jpg(path: Path) -> Path:
    """Write a tiny JPEG.

    Args:
        path: Destination.

    Returns:
        The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path)
    return path


def _dataset(root: Path) -> Path:
    """Build a dataset like a real survey: two flights, logs, a duplicated video, ignored folders.

    Args:
        root: Dataset folder.

    Returns:
        The folder.
    """
    for i in range(3):
        _jpg(root / "Flight A" / f"DJI_{i:04d}.JPG")
        _jpg(root / "flight_b" / f"DJI_{i:04d}.JPG")          # same names as flight A
        _jpg(root / "ignore" / "old" / f"x_{i}.jpg")
        _jpg(root / "Flight A" / "results" / f"r_{i}.jpg")    # nested ignored folder
    (root / "Logs").mkdir()
    (root / "Logs" / "targets.csv").write_text(TARGETS)
    (root / "Logs" / "targets_SC.csv").write_text(TARGETS_SC)
    (root / "Logs" / "app-log.txt").write_text("e: 2026-11-19 12:05:20 @ Active init\n")
    (root / "Flight A" / "targets.csv").write_text(TARGETS)    # copy of the same targets
    (root / "Video").mkdir()
    (root / "Video" / "clip.MOV").write_bytes(b"\x00" * 1000 + b"video")
    shutil.copy(root / "Video" / "clip.MOV", root / "flight_b" / "clip.MOV")
    (root / "results").mkdir()
    (root / "results" / "Orto.tif").write_bytes(b"tif")
    return root


def test_scan_dataset(tmp_path: Path) -> None:
    """Photo groups per folder, ignored folders (any depth), one copy of a duplicated video, side files.

    Args:
        tmp_path: Pytest temporary directory.
    """
    ds = scan_dataset(_dataset(tmp_path))
    assert [(g.label, len(g.photos)) for g in ds.groups] == [("Flight A", 3), ("flight_b", 3)]
    assert [v.name for v in ds.videos] == ["clip.MOV"]
    assert {p.name for p in ds.side_files} == {"targets.csv", "targets_SC.csv", "app-log.txt"}
    assert {p.name for p in ds.ignored} == {"ignore", "results"}
    assert not ds.flat and scan_dataset(tmp_path / "flight_b").groups[0].label == "."


def test_gcp_layouts_and_merge(tmp_path: Path) -> None:
    """``id,x,y,z,name`` and ``id,x,y,z`` layouts; copies in several folders merge into one set of named points.

    Args:
        tmp_path: Pytest temporary directory.
    """
    root = _dataset(tmp_path)
    assert list(read_gcps(root / "Logs" / "targets.csv")) == ["D1", "D2", "D3"]
    assert list(read_gcps(root / "Logs" / "targets_SC.csv")) == ["1", "2", "3"]
    files = find_gcp_files(root)
    assert {p.name for p in files} == {"targets.csv", "targets_SC.csv"} and len(files) == 3
    merged, _ = read_gcp_files(files)
    assert sorted(merged) == ["D1", "D2", "D3"]
    assert np.allclose(merged["D2"], [411345.289, 4496323.242, 947.590])


def test_dji_srt(tmp_path: Path) -> None:
    """Both DJI subtitle layouts give per-frame GPS.

    Args:
        tmp_path: Pytest temporary directory.
    """
    srt = tmp_path / "DJI_0001.SRT"
    srt.write_text("1\n00:00:00,000 --> 00:00:00,033\n<font>FrameCnt: 1 [latitude: 40.613379] "
                   "[longitude: -4.048193] [rel_alt: 21.700 abs_alt: 965.123]</font>\n\n"
                   "2\n00:00:01,000 --> 00:00:01,033\nHOME(-4.0, 40.6) GPS(-4.048100, 40.613400, 21)\n")
    track = read_dji_srt(srt)
    assert len(track) == 2
    assert track[0][1:] == (40.613379, -4.048193, 965.123)
    assert track[1][0] == 1.0 and track[1][1:3] == (40.6134, -4.0481)


def test_extract_frames(tmp_path: Path) -> None:
    """Evenly spaced, downscaled frames are extracted from a video.

    Args:
        tmp_path: Pytest temporary directory.
    """
    try:
        ffmpeg = ffmpeg_executable()
    except RuntimeError:
        pytest.skip("ffmpeg not available")
    video = tmp_path / "test.mp4"
    subprocess.run([ffmpeg, "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=2",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    frames = extract_frames(video, tmp_path / "frames", num_frames=4, max_size=320)
    assert len(frames) == 4
    assert [round(t, 2) for _, t in frames] == [0.25, 0.75, 1.25, 1.75]
    assert Image.open(frames[0][0]).size == (320, 180)
