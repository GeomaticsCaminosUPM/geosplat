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


PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def unzip_images(zip_path: str | Path, out_dir: str | Path | None = None) -> Path:
    """Extract the photos (and CSV files, e.g. GCPs) of a zip into a flat folder.

    Subfolders are flattened; ``__MACOSX`` and hidden files are skipped. If the
    output folder already contains photos, nothing is extracted again.

    Args:
        zip_path: Zip archive.
        out_dir: Destination folder. Default: ``images/`` next to the zip if the
            zip is called ``images.zip``, otherwise a folder named like the zip.

    Returns:
        The folder with the photos.
    """
    zip_path = Path(zip_path)
    if out_dir is None:
        out_dir = zip_path.parent / ("images" if zip_path.stem.lower() == "images" else zip_path.stem)
    out_dir = Path(out_dir)
    if out_dir.is_dir() and any(p.suffix.lower() in PHOTO_EXTENSIONS for p in out_dir.iterdir()):
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts
            if info.is_dir() or "__MACOSX" in parts or any(p.startswith(".") for p in parts):
                continue
            if Path(info.filename).suffix.lower() not in PHOTO_EXTENSIONS + (".csv", ".txt"):
                continue
            target = out_dir / Path(info.filename).name
            if target.exists():  # same name in another subfolder
                target = out_dir / f"{Path(info.filename).parent.name}_{Path(info.filename).name}"
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    logger.info("Extracted %d files from %s to %s", count, zip_path.name, out_dir)
    return out_dir


def resolve_image_source(path: str | Path) -> Path:
    """Turn a user-given photo source into a folder of photos.

    Accepts a folder of photos, a dataset folder with ``images/``, a ``.zip``
    of photos, or a dataset folder containing a single zip (and no photos),
    which is extracted to ``images/``.

    Args:
        path: Folder or zip.

    Returns:
        Folder containing the photos.
    """
    path = Path(path).expanduser()
    if path.is_file() and path.suffix.lower() == ".zip":
        return unzip_images(path)
    if path.is_dir():
        if (path / "images").is_dir():
            return resolve_image_source(path / "images")
        has_photos = any(p.suffix.lower() in PHOTO_EXTENSIONS for p in path.iterdir())
        zips = sorted(path.glob("*.zip")) + sorted(path.glob("*.ZIP"))
        if not has_photos and len(zips) == 1:
            return unzip_images(zips[0], path if path.name == "images" else path / "images")
    elif path.with_suffix(".zip").is_file():  # "flight" given, "flight.zip" exists
        return unzip_images(path.with_suffix(".zip"), path)
    return path
