"""Input datasets: any folder tree with photos, videos and side files.

A dataset is a folder. Every non-ignored subfolder that contains photos is a
*photo group*; every video is a *video source*; all other files (CSV, TXT,
GeoPackage, MRK, GPX, SRT, XMP...) are *side files* that may hold photo
positions or ground control points. Folders such as ``ignore``, ``results`` or
``output*`` are skipped, and identical videos found in several folders are used
once.

For SfM each group becomes a subfolder of the prepared images (one COLMAP
camera per group), and a ``catalogue.json`` maps every prepared image back to
its original photo or video frame. :func:`dataset_positions` uses it to give
each prepared image its position.
"""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .sfm import IMAGE_EXTENSIONS

logger = logging.getLogger("geosplat")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".m2ts", ".3gp", ".webm", ".lrv"}
SIDE_EXTENSIONS = {".csv", ".txt", ".gpkg", ".geojson", ".json", ".shp", ".kml", ".fgb", ".gml", ".mrk", ".gpx", ".srt",
                   ".xmp", ".obs", ".pos"}
DEFAULT_IGNORE = ("ignore", "ignored", "exclude", "excluded", "results", "result", "output*", "outputs",
                  "colmap", "__macosx", ".*", "_*", "thumbnails", "thumbs")


@dataclass
class PhotoGroup:
    """Photos of one folder (one camera in SfM).

    Attributes:
        label: Folder path relative to the dataset root (``"."`` for the root).
        folder: Folder path.
        photos: Photo paths, sorted by name.
    """

    label: str
    folder: Path
    photos: list[Path]


@dataclass
class Dataset:
    """Everything usable found under a dataset folder.

    Attributes:
        root: Dataset folder.
        groups: Photo groups.
        videos: Video files (duplicates removed).
        side_files: Other files that may hold positions or ground control points.
        ignored: Folders that were skipped.
    """

    root: Path
    groups: list[PhotoGroup] = field(default_factory=list)
    videos: list[Path] = field(default_factory=list)
    side_files: list[Path] = field(default_factory=list)
    ignored: list[Path] = field(default_factory=list)

    @property
    def photos(self) -> list[Path]:
        """All photos of all groups.

        Returns:
            Photo paths.
        """
        return [p for g in self.groups for p in g.photos]

    @property
    def flat(self) -> bool:
        """Whether the dataset is a single folder of photos (no subfolders, no videos).

        Returns:
            ``True`` for the classic layout.
        """
        return len(self.groups) == 1 and self.groups[0].label == "." and not self.videos

    def summary(self) -> str:
        """One-line description of the dataset.

        Returns:
            Human-readable summary.
        """
        parts = [f"{len(self.photos)} photos in {len(self.groups)} folder(s)"]
        if self.videos:
            parts.append(f"{len(self.videos)} video(s)")
        if self.side_files:
            parts.append(f"{len(self.side_files)} side file(s)")
        if self.ignored:
            parts.append("ignored: " + ", ".join(str(p.relative_to(self.root)) for p in self.ignored))
        return "; ".join(parts)


def _ignored(name: str, patterns: tuple[str, ...]) -> bool:
    """Whether a folder name matches an ignore pattern (case-insensitive).

    Args:
        name: Folder name.
        patterns: ``fnmatch`` patterns.

    Returns:
        ``True`` if the folder must be skipped.
    """
    return any(fnmatch.fnmatch(name.lower(), p.lower()) for p in patterns)


def _fingerprint(path: Path, chunk: int = 4 << 20) -> str:
    """Cheap content fingerprint (size + first and last 4 MB) to detect duplicate videos.

    Args:
        path: File.
        chunk: Bytes read at each end.

    Returns:
        Hex digest.
    """
    size = path.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(size - chunk, chunk))
            h.update(f.read(chunk))
    return h.hexdigest()


def _tree_stamp(root: Path) -> tuple:
    """Modification fingerprint of a folder tree (folder mtimes change when files are added or removed).

    Args:
        root: Folder.

    Returns:
        Hashable fingerprint.
    """
    import os

    return tuple((d, os.stat(d).st_mtime_ns) for d, _, _ in os.walk(root))


def scan_dataset(root: str | Path, ignore: tuple[str, ...] | list[str] | None = None) -> Dataset:
    """Find photos, videos and side files under a folder (recursively).

    Results are cached until a folder of the tree changes.

    Args:
        root: Dataset folder.
        ignore: Extra folder-name patterns to skip (added to :data:`DEFAULT_IGNORE`).

    Returns:
        The :class:`Dataset`.

    Raises:
        FileNotFoundError: If ``root`` is not a folder.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    return _scan_cached(str(root), tuple(ignore or ()), _tree_stamp(root))


@functools.lru_cache(maxsize=16)
def _scan_cached(root_str: str, ignore: tuple[str, ...], stamp: tuple) -> Dataset:
    """Cached implementation of :func:`scan_dataset`.

    Args:
        root_str: Resolved dataset folder.
        ignore: Extra ignore patterns.
        stamp: Tree fingerprint (invalidates the cache).

    Returns:
        The :class:`Dataset`.
    """
    root = Path(root_str)
    patterns = tuple(DEFAULT_IGNORE) + ignore
    ds = Dataset(root)
    seen_videos: dict[str, Path] = {}

    def walk(folder: Path) -> None:
        """Visit a folder and its non-ignored subfolders.

        Args:
            folder: Folder to visit.
        """
        entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
        photos = [p for p in entries if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if photos:
            label = "." if folder == root else str(folder.relative_to(root))
            ds.groups.append(PhotoGroup(label, folder, photos))
        for p in entries:
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                fp = _fingerprint(p)
                if fp in seen_videos:
                    _log_once("Skipping %s (same video as %s)", p.relative_to(root), seen_videos[fp].relative_to(root))
                else:
                    seen_videos[fp] = p
                    ds.videos.append(p)
            elif p.is_file() and p.suffix.lower() in SIDE_EXTENSIONS:
                ds.side_files.append(p)
            elif p.is_dir():
                if _ignored(p.name, patterns):
                    ds.ignored.append(p)
                else:
                    walk(p)

    walk(root)
    return ds


def list_photos(root: str | Path, ignore: tuple[str, ...] | list[str] | None = None) -> list[Path]:
    """All photos of a dataset folder (recursively, ignored folders skipped).

    Args:
        root: Dataset folder.
        ignore: Extra folder-name patterns to skip.

    Returns:
        Photo paths.
    """
    return scan_dataset(root, ignore).photos


def group_slug(label: str) -> str:
    """Folder-safe name for a photo group or video.

    Args:
        label: Group label or video path.

    Returns:
        Slug (letters, digits, ``-`` and ``_``).
    """
    slug = re.sub(r"[^\w\-]+", "_", label.replace("/", "__")).strip("_")
    return slug or "root"


def ffmpeg_executable() -> str:
    """Path of an ffmpeg executable (system ffmpeg, else the one bundled with ``imageio-ffmpeg``).

    Returns:
        Executable path.

    Raises:
        RuntimeError: If no ffmpeg is available.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Videos need ffmpeg: install it or `pip install imageio-ffmpeg`.") from exc


def video_duration(path: str | Path) -> float:
    """Duration of a video in seconds.

    Args:
        path: Video file.

    Returns:
        Duration (``0.0`` if it cannot be read).
    """
    out = subprocess.run([ffmpeg_executable(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out.stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0


def extract_frames(video: str | Path, out_dir: str | Path, num_frames: int, max_size: int = 0) -> list[tuple[Path, float]]:
    """Extract evenly spaced frames from a video as JPEG files.

    Frames are taken in the middle of ``num_frames`` equal time slots. Each one
    is read by seeking to its time (only that frame is decoded, so long 4K
    videos stay cheap) and downscaled when ``max_size`` is set.

    Args:
        video: Video file.
        out_dir: Destination folder.
        num_frames: Number of frames to extract.
        max_size: Maximum longest side in pixels (``0`` keeps the video size).

    Returns:
        List of ``(frame_path, time_seconds)``.
    """
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = video_duration(video)
    if duration <= 0 or num_frames <= 0:
        return []
    scale = []
    if max_size:
        scale = ["-vf", f"scale='if(gt(iw,ih),min({max_size},iw),-2)':'if(gt(iw,ih),-2,min({max_size},ih))'"]
    out = []
    for i in range(num_frames):
        t = (i + 0.5) * duration / num_frames
        frame = out_dir / f"{group_slug(video.stem)}_{i + 1:05d}.jpg"
        subprocess.run([ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}",
                        "-i", str(video), "-frames:v", "1", *scale, "-q:v", "2", str(frame)], check=True)
        if frame.is_file():
            out.append((frame, t))
    return out


def read_dji_srt(path: str | Path) -> list[tuple[float, float, float, float]]:
    """Read the per-frame GPS of a DJI subtitle (``.SRT``) file.

    Supports the common DJI layouts (``[latitude: x] [longitude: y] ... abs_alt: z``
    and ``GPS(lon, lat, alt)``).

    Args:
        path: SRT file.

    Returns:
        List of ``(time_s, lat, lon, alt)`` sorted by time (empty if no GPS).
    """
    text = Path(path).read_text(errors="replace")
    out = []
    for block in re.split(r"\n\s*\n", text):
        t = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->", block)
        if not t:
            continue
        ts = int(t.group(1)) * 3600 + int(t.group(2)) * 60 + int(t.group(3)) + int(t.group(4)) / 1000
        lat = re.search(r"latitude\s*:\s*([-+]?\d+\.\d+)", block)
        lon = re.search(r"longitude\s*:\s*([-+]?\d+\.\d+)", block)
        alt = re.search(r"abs_alt\s*:\s*([-+]?\d+(?:\.\d+)?)", block) or re.search(r"altitude\s*:\s*([-+]?\d+(?:\.\d+)?)", block)
        if not (lat and lon):
            g = re.search(r"GPS\s*\(\s*([-+]?\d+\.\d+)\s*,\s*([-+]?\d+\.\d+)\s*,\s*([-+]?\d+(?:\.\d+)?)", block)
            if not g:
                continue
            out.append((ts, float(g.group(2)), float(g.group(1)), float(g.group(3))))
            continue
        out.append((ts, float(lat.group(1)), float(lon.group(1)), float(alt.group(1)) if alt else float("nan")))
    return sorted(out)


def _srt_for(video: Path, side_files: list[Path]) -> Path | None:
    """Find the DJI subtitle file of a video (same stem, ``.srt``).

    Args:
        video: Video file.
        side_files: Dataset side files.

    Returns:
        SRT path or ``None``.
    """
    for p in [video.with_suffix(".SRT"), video.with_suffix(".srt"), *side_files]:
        if p.is_file() and p.suffix.lower() == ".srt" and p.stem.lower() == video.stem.lower():
            return p
    return None


def catalogue_key(name: str | Path) -> str:
    """Key of a prepared image (relative path without extension, lower case).

    Args:
        name: Prepared image name as COLMAP stores it (e.g. ``"group/DJI_0001.jpg"``).

    Returns:
        Matching key.
    """
    return str(Path(name).with_suffix("")).replace("\\", "/").lower()


def dataset_positions(work_dir: str | Path, config: object | None = None) -> dict:
    """Positions of the prepared images of an SfM run, from the catalogue.

    Photos get their position from :func:`geosplat.positions.photo_positions`
    (EXIF, sidecars, positions files anywhere in the dataset); video frames
    from the video's DJI ``.SRT`` file, if any.

    Args:
        work_dir: SfM work folder containing ``catalogue.json``.
        config: Optional :class:`geosplat.geo.GeoConfig` (positions file / CRS / GPX offset).

    Returns:
        ``{catalogue_key: PhotoPosition}``.
    """
    from .positions import PhotoPosition, photo_positions, position_of

    cat_file = Path(work_dir) / "catalogue.json"
    if not cat_file.is_file():
        return {}
    cat = json.loads(cat_file.read_text())
    side = [Path(p) for p in cat.get("side_files", [])]
    positions_file = getattr(config, "positions", None)
    crs = getattr(config, "positions_crs", None)
    gpx_offset = getattr(config, "gpx_time_offset", None)
    by_folder: dict[str, dict] = {}
    srt_cache: dict[str, list] = {}
    out = {}
    for rel, entry in cat["images"].items():
        src = Path(entry["source"])
        if entry["kind"] == "photo":
            folder = str(src.parent)
            if folder not in by_folder:
                by_folder[folder] = photo_positions(src.parent, positions_file, crs, gpx_offset, extra_files=side)
            pos = position_of(src, by_folder[folder])
            if pos is not None:
                out[catalogue_key(rel)] = pos
        elif entry["kind"] == "video_frame" and entry.get("srt"):
            track = srt_cache.setdefault(entry["srt"], read_dji_srt(entry["srt"]))
            if track:
                tt = np.array([t[0] for t in track])
                t = float(entry["time"])
                if tt[0] - 1 <= t <= tt[-1] + 1:
                    vals = [float(np.interp(t, tt, [r[k] for r in track])) for k in (1, 2, 3)]
                    out[catalogue_key(rel)] = PhotoPosition(vals[0], vals[1], vals[2], source=Path(entry["srt"]).name)
    return out


def photo_positions_in(root: str | Path, config: object | None = None,
                       ignore: tuple[str, ...] | list[str] | None = None) -> dict[Path, object]:
    """Positions of every photo of a dataset folder, keyed by original photo path.

    Args:
        root: Dataset folder (or a plain folder of photos).
        config: Optional :class:`geosplat.geo.GeoConfig`.
        ignore: Extra folder-name patterns to skip.

    Returns:
        ``{photo_path: PhotoPosition}`` for photos with a known position.
    """
    from .positions import photo_positions, position_of

    ds = scan_dataset(root, ignore)
    out = {}
    for g in ds.groups:
        pos = photo_positions(g.folder, getattr(config, "positions", None), getattr(config, "positions_crs", None),
                              getattr(config, "gpx_time_offset", None), extra_files=ds.side_files)
        for p in g.photos:
            q = position_of(p, pos)
            if q is not None:
                out[p] = q
    return out


_LOGGED: set[str] = set()


def _log_once(msg: str, *args: object) -> None:
    """Log an info message only the first time it appears (datasets are scanned several times per run).

    Args:
        msg: Message format.
        *args: Format arguments.
    """
    text = msg % args
    if text not in _LOGGED:
        _LOGGED.add(text)
        logger.info(text)


def resolve_subfolder(root: str | Path, folder: str | Path) -> Path:
    """Resolve a folder given as an absolute path or relative to the dataset folder.

    Args:
        root: Dataset folder.
        folder: Absolute path, or path / folder name relative to ``root``.

    Returns:
        Resolved folder.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    p = Path(folder).expanduser()
    for cand in ((p,) if p.is_absolute() else (Path(root) / p, p)):   # relative: dataset folder first
        if cand.is_dir():
            return cand.resolve()
    raise FileNotFoundError(f"Folder {folder!r} not found (absolute or inside {root})")


def is_within(path: str | Path, folder: str | Path) -> bool:
    """Whether a path lies inside a folder.

    Args:
        path: File path.
        folder: Folder.

    Returns:
        ``True`` if ``path`` is inside ``folder`` (at any depth).
    """
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
        return True
    except ValueError:
        return False


def catalogue_sources(work_dir: str | Path) -> dict[str, Path]:
    """Original file of each prepared image of an SfM run.

    Args:
        work_dir: SfM work folder containing ``catalogue.json``.

    Returns:
        ``{catalogue_key: original photo or video path}`` (empty without catalogue).
    """
    cat_file = Path(work_dir) / "catalogue.json"
    if not cat_file.is_file():
        return {}
    cat = json.loads(cat_file.read_text())
    return {catalogue_key(rel): Path(e["source"]) for rel, e in cat["images"].items()}


def cameras_in(scene: object, work_dir: str | Path, folder: str | Path) -> list:
    """Cameras of a scene whose original photo is inside ``folder``.

    Args:
        scene: :class:`geosplat.scene.Scene`.
        work_dir: SfM work folder (with ``catalogue.json``).
        folder: Folder whose photos are wanted.

    Returns:
        The matching cameras.
    """
    sources = catalogue_sources(work_dir)
    return [c for c in scene.cameras
            if catalogue_key(c.name) in sources and is_within(sources[catalogue_key(c.name)], folder)]


def high_res_cameras(work_dir: str | Path, cameras: list, out_dir: str | Path) -> list:
    """Undistorted full-resolution versions of cameras whose original is a photo.

    SfM runs on downscaled photos; for a sharper orthophoto, the original photo
    is undistorted with the SfM lens model (rescaled to its size). Poses are
    unchanged. Cameras from video frames (or without catalogue entry) are
    returned as they are. Results are cached in ``out_dir``.

    Args:
        work_dir: SfM work folder (``catalogue.json`` and ``sparse/best``).
        cameras: Cameras of the (undistorted, normalized) scene.
        out_dir: Folder for the full-resolution undistorted images.

    Returns:
        Cameras at full resolution (same order).
    """
    import numpy as np
    import pycolmap
    from PIL import Image, ImageOps

    from .scene import Camera

    work_dir, out_dir = Path(work_dir), Path(out_dir)
    best, cat_file = work_dir / "sparse" / "best", work_dir / "catalogue.json"
    if not best.is_dir() or not cat_file.is_file():
        return list(cameras)
    cat = json.loads(cat_file.read_text())["images"]
    meta_file = out_dir / "cameras.json"
    meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
    rec = None
    out_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for cam in cameras:
        entry = cat.get(cam.name)
        if entry is None or entry["kind"] != "photo":
            result.append(cam)
            continue
        dst = out_dir / (cam.name.replace("/", "__").rsplit(".", 1)[0] + ".jpg")
        m = meta.get(cam.name)
        if m is None or not dst.is_file() or m.get("source") != entry["source"]:
            if rec is None:
                rec = pycolmap.Reconstruction(str(best))
                by_name = {im.name: im for im in rec.images.values()}
            im = by_name.get(cam.name)
            if im is None:
                result.append(cam)
                continue
            with Image.open(entry["source"]) as orig:
                arr = np.asarray(ImageOps.exif_transpose(orig).convert("RGB"))
            c = rec.cameras[im.camera_id]
            dist = pycolmap.Camera(model=c.model, width=c.width, height=c.height, params=c.params)
            dist.rescale(arr.shape[1], arr.shape[0])
            bitmap, und = pycolmap.undistort_image(pycolmap.UndistortCameraOptions(),
                                                   pycolmap.Bitmap.from_array(arr), dist)
            Image.fromarray(bitmap.to_array()).save(dst, quality=95)
            m = {"source": entry["source"], "K": und.calibration_matrix().tolist(),
                 "width": und.width, "height": und.height}
            meta[cam.name] = m
            meta_file.write_text(json.dumps(meta, indent=1))
        result.append(Camera(cam.name, dst, np.array(m["K"], dtype=np.float64), cam.world_to_cam.copy(),
                             int(m["width"]), int(m["height"])))
    return result
