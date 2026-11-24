"""Download helpers for public sample datasets."""

from __future__ import annotations

import io
import logging
import math
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm.auto import tqdm

logger = logging.getLogger("geosplat")

SOUTH_BUILDING_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip"
SENECA_API = "https://api.github.com/repos/OpenDroneMap/odm_data_seneca/contents/images"
# Agisoft Metashape sample "Aerial images with GCPs" (data by GeoScan): 444 nadir photos (Sony RX1R),
# camera positions and 17 surveyed ground control points, WGS84 / ellipsoidal heights.
AERIAL_GCPS_URL = "https://download.agisoft.com/datasets/aerial_images_with_gcps.zip"
AERIAL_GCPS_BLOCK = ("4", "5", "9")      # GCPs at the centre of the downloaded photo block


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
    """Extract a zip of photos (and videos, positions / GCP files), keeping its folders.

    The folder structure is kept (several flights stay separate); a single
    top-level folder wrapping everything is dropped. ``__MACOSX`` and hidden
    files are skipped, and nothing is extracted again if the destination already
    has content.

    Args:
        zip_path: Zip archive.
        out_dir: Destination folder. Default: ``images/`` next to the zip if the
            zip is called ``images.zip``, otherwise a folder named like the zip.

    Returns:
        The extracted dataset folder.
    """
    from .sources import SIDE_EXTENSIONS, VIDEO_EXTENSIONS

    zip_path = Path(zip_path)
    if out_dir is None:
        out_dir = zip_path.parent / ("images" if zip_path.stem.lower() == "images" else zip_path.stem)
    out_dir = Path(out_dir)
    if out_dir.is_dir() and any(out_dir.iterdir()):
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = set(PHOTO_EXTENSIONS) | VIDEO_EXTENSIONS | SIDE_EXTENSIONS
    with zipfile.ZipFile(zip_path) as zf:
        members = [i for i in zf.infolist() if not i.is_dir()
                   and "__MACOSX" not in Path(i.filename).parts
                   and not any(p.startswith(".") for p in Path(i.filename).parts)
                   and Path(i.filename).suffix.lower() in keep]
        tops = {Path(i.filename).parts[0] for i in members}
        strip = len(tops) == 1 and all(len(Path(i.filename).parts) > 1 for i in members)
        for info in members:
            parts = Path(info.filename).parts[1:] if strip else Path(info.filename).parts
            target = out_dir.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    logger.info("Extracted %d files from %s to %s", len(members), zip_path.name, out_dir)
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
        from .sources import scan_dataset

        has_content = bool(scan_dataset(path).photos or scan_dataset(path).videos)
        zips = sorted(path.glob("*.zip")) + sorted(path.glob("*.ZIP"))
        if not has_content and len(zips) == 1:
            return unzip_images(zips[0], path if path.name == "images" else path / "images")
    elif path.with_suffix(".zip").is_file():  # "flight" given, "flight.zip" exists
        return unzip_images(path.with_suffix(".zip"), path)
    return path


class _RemoteFile(io.RawIOBase):
    """Read-only, seekable view of a URL through HTTP range requests (to read single files of a large zip)."""

    def __init__(self, url: str) -> None:
        """Open the URL.

        Args:
            url: File URL; the server must support ``Range`` requests.
        """
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
            self.size = int(resp.headers["Content-Length"])

    def readable(self) -> bool:
        """File is readable.

        Returns:
            ``True``.
        """
        return True

    def seekable(self) -> bool:
        """File is seekable.

        Returns:
            ``True``.
        """
        return True

    def tell(self) -> int:
        """Current position.

        Returns:
            Byte offset.
        """
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        """Move the read position.

        Args:
            offset: Byte offset.
            whence: 0 = from start, 1 = from current, 2 = from end.

        Returns:
            New position.
        """
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def readinto(self, buf: bytearray) -> int:
        """Read bytes at the current position.

        Args:
            buf: Buffer to fill.

        Returns:
            Number of bytes read.
        """
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def download_aerial_gcps(images_dir: str | Path, num_images: int | None = 40) -> Path:
    """Download a block of the Agisoft "Aerial images with GCPs" sample (data by GeoScan).

    Only the needed files are read from the 15 GB zip (HTTP range requests):
    the ``num_images`` photos nearest to GCPs 4, 5 and 9 (about 11 MB each),
    the camera positions table ``Cameras_WGS84.txt`` (the photos have no GPS
    EXIF) and ``GCPs_WGS84.txt`` with the GCPs inside the block.

    Args:
        images_dir: Destination folder.
        num_images: Number of photos (``None``: 40).

    Returns:
        ``images_dir``.
    """
    images_dir = Path(images_dir)
    num_images = num_images or 40
    if len(list(images_dir.glob("*.JPG"))) >= num_images:
        return images_dir
    images_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Reading the index of %s", AERIAL_GCPS_URL)
    z = zipfile.ZipFile(io.BufferedReader(_RemoteFile(AERIAL_GCPS_URL), 1 << 20))
    base = "aerial_images_with_gcps/"
    cams_txt = z.read(base + "Metadata/Cameras_WGS84.txt").decode("utf-8", "replace")
    gcps_txt = z.read(base + "Metadata/GCPs_WGS84.txt").decode("utf-8", "replace")
    rows = [ln.split("\t") for ln in cams_txt.splitlines() if ln.strip() and not ln.startswith("#")]
    cams = {r[0].strip(): (float(r[1]), float(r[2])) for r in rows}
    gcps = {r[0].strip(): (float(r[1]), float(r[2]))
            for r in (ln.split("\t") for ln in gcps_txt.splitlines() if ln.strip() and not ln.startswith("#"))}
    lat0 = sum(gcps[g][0] for g in AERIAL_GCPS_BLOCK) / len(AERIAL_GCPS_BLOCK)
    lon0 = sum(gcps[g][1] for g in AERIAL_GCPS_BLOCK) / len(AERIAL_GCPS_BLOCK)
    k = math.cos(math.radians(lat0))
    in_zip = {Path(i.filename).name for i in z.infolist()}
    names = sorted((n for n in cams if n in in_zip),
                   key=lambda n: math.hypot(cams[n][0] - lat0, (cams[n][1] - lon0) * k))[:num_images]
    lats = [cams[n][0] for n in names]
    lons = [cams[n][1] for n in names]
    keep = [g for g, (la, lo) in gcps.items() if min(lats) <= la <= max(lats) and min(lons) <= lo <= max(lons)]
    (images_dir / "Cameras_WGS84.txt").write_text(cams_txt)
    header = next(ln for ln in gcps_txt.splitlines() if ln.startswith("#"))
    body = [ln for ln in gcps_txt.splitlines() if ln.split("\t")[0].strip() in keep]
    (images_dir / "GCPs_WGS84.txt").write_text("\n".join([header, *body]) + "\n")
    for n in tqdm(names, desc="aerial_gcps photos"):
        if not (images_dir / n).is_file():
            with z.open(base + "Images/" + n) as src, open(images_dir / n, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    logger.info("Downloaded %d photos and GCPs %s to %s", len(names), keep, images_dir)
    return images_dir


SAMPLES = {"seneca": download_seneca, "south_building": download_south_building, "aerial_gcps": download_aerial_gcps}


def prepare_images(source: str | Path, examples_dir: str | Path = "examples", num_images: int = 40) -> Path:
    """Turn a notebook/CLI photo source into a folder of photos.

    ``source`` may be a folder of photos, a dataset folder with ``images/``, a
    ``.zip`` of photos (extracted next to it), or the name of a public sample
    (``"seneca"``: drone flight with GPS; ``"south_building"``: hand-held photos;
    ``"aerial_gcps"``: survey flight with a positions file and ground control points),
    which is downloaded into ``<examples_dir>/<name>/images`` on first use.

    Args:
        source: Path or sample name.
        examples_dir: Folder for downloaded samples.
        num_images: Number of sample photos to download.

    Returns:
        Folder containing the photos.

    Raises:
        FileNotFoundError: If ``source`` is neither an existing path nor a sample name.
    """
    text = str(source).strip()
    path = Path(text).expanduser()
    if not path.exists() and not path.with_suffix(".zip").exists():
        if text in SAMPLES:
            images = Path(examples_dir) / text / "images"
            SAMPLES[text](images, num_images=num_images)   # skipped if already downloaded
            return images
        raise FileNotFoundError(
            f"{text!r} is not an existing folder or .zip, nor a sample ({', '.join(SAMPLES)})")
    return resolve_image_source(path)
