"""Download helpers for public sample datasets."""

from __future__ import annotations

import logging
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm.auto import tqdm

logger = logging.getLogger("geosplat")

SOUTH_BUILDING_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip"
SENECA_API = "https://api.github.com/repos/OpenDroneMap/odm_data_seneca/contents/images"


def download_file(url: str, dest: str | Path) -> Path:
    """Download a URL to a file, with a progress bar.

    Args:
        url: Source URL.
        dest: Destination file path.

    Returns:
        The destination path.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def download_south_building(
    images_dir: str | Path, num_images: int | None = 40, start: int = 0, stride: int = 1
) -> Path:
    """Download COLMAP's *South Building* dataset (128 photos, ~420 MB).

    Only the images are kept (the bundled reconstruction is discarded) so the
    full images -> SfM -> splat pipeline can be exercised.

    Args:
        images_dir: Folder where the images are written.
        num_images: Number of images to keep (``None`` keeps all 128). Fewer
            images make SfM and training faster.
        start: Index of the first image to keep (images are sorted by name).
        stride: Keep every ``stride``-th image.

    Returns:
        The images folder.
    """
    images_dir = Path(images_dir)
    existing = [p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if existing:
        logger.info("%s already has %d images; skipping download", images_dir, len(existing))
        return images_dir
    images_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = download_file(SOUTH_BUILDING_URL, Path(tmp) / "south-building.zip")
        with zipfile.ZipFile(zip_path) as zf:
            names = sorted(n for n in zf.namelist() if "/images/" in n and n.lower().endswith(".jpg"))
            names = names[start::stride]
            if num_images is not None:
                names = names[:num_images]
            for name in tqdm(names, desc="Extracting"):
                with zf.open(name) as src, open(images_dir / Path(name).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    return images_dir


def download_seneca(images_dir: str | Path, num_images: int | None = 40, start: int = 0, stride: int = 1) -> Path:
    """Download photos of OpenDroneMap's *Seneca* drone dataset (GPS-tagged, 167 photos).

    Photos are fetched one by one (about 2.3 MB each), so only the selected
    subset is downloaded.

    Args:
        images_dir: Folder where the images are written.
        num_images: Number of images to keep (``None`` keeps all 167).
        start: Index of the first image (sorted by name, i.e. flight order).
        stride: Keep every ``stride``-th image.

    Returns:
        The images folder.
    """
    import json

    images_dir = Path(images_dir)
    existing = [p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg"}]
    if existing:
        logger.info("%s already has %d images; skipping download", images_dir, len(existing))
        return images_dir
    images_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(SENECA_API, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as resp:
        listing = json.load(resp)
    files = sorted((f for f in listing if f["name"].lower().endswith((".jpg", ".jpeg"))), key=lambda f: f["name"])
    files = files[start::stride]
    if num_images is not None:
        files = files[:num_images]
    for f in tqdm(files, desc="Downloading Seneca photos"):
        with urllib.request.urlopen(f["download_url"]) as resp, open(images_dir / f["name"], "wb") as out:
            shutil.copyfileobj(resp, out)
    return images_dir
