"""Structure-from-Motion with pycolmap: images -> camera poses + sparse points."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pycolmap
from PIL import Image, ImageOps

logger = logging.getLogger("geosplat")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SfMConfig:
    """Options for the COLMAP Structure-from-Motion stage.

    Attributes:
        max_image_size: Images are downscaled so their longest side is at most
            this many pixels before SfM (and training). ``0`` keeps originals.
        matcher: ``"auto"`` picks ``"spatial"`` for GPS-tagged images and
            ``"exhaustive"`` otherwise. ``"exhaustive"`` matches every image pair
            (unordered sets up to a few hundred images); ``"sequential"`` matches
            neighbouring frames (video); ``"spatial"`` matches only images whose
            GPS positions are close (drone flights; scales to thousands of images).
        spatial_neighbors: Number of nearest GPS neighbours matched by the spatial matcher.
        sequential_overlap: Number of neighbours matched by the sequential matcher.
        mapper: ``"incremental"`` (classic COLMAP, most robust) or ``"global"``
            (GLOMAP, usually much faster on large image sets).
        camera_model: COLMAP camera model estimated during SfM
            (e.g. ``"SIMPLE_RADIAL"``, ``"OPENCV"``, ``"PINHOLE"``).
        single_camera: Assume every image was taken with the same camera and
            zoom level. Improves robustness for typical captures.
        max_num_features: Maximum SIFT features per image.
        use_gpu: Use the GPU for feature extraction, matching and bundle
            adjustment when pycolmap was built with CUDA (``pycolmap-cuda12``,
            installed by default on Linux x86_64).
        overwrite: Re-run SfM even if a previous result exists in the work dir.
        max_images: Use at most this many images, evenly spaced over the
            (name-sorted) sequence. ``None`` uses all.
        image_stride: Use every ``image_stride``-th image.
        polygon: Vector file (``.gpkg``, ``.shp``, ``.geojson``...) with an area
            of interest; only GPS-tagged images taken inside it are used.
        polygon_buffer: Grow the polygon by this many meters before filtering.
        image_list: Only use these photos: a list of file names (or paths relative
            to the dataset folder), or a text file with one per line (e.g. saved by
            the notebook photo map).
        video_frames: Frames extracted from each video found in the dataset.
        ignore_folders: Extra folder-name patterns to skip when scanning the
            dataset (``ignore``, ``results``, ``output*``... are always skipped).
    """

    max_image_size: int = 1600
    matcher: Literal["auto", "exhaustive", "sequential", "spatial"] = "auto"
    sequential_overlap: int = 10
    spatial_neighbors: int = 30
    mapper: Literal["incremental", "global"] = "incremental"
    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = True
    max_num_features: int = 8192
    use_gpu: bool = True
    overwrite: bool = False
    max_images: int | None = None
    image_stride: int = 1
    polygon: str | Path | None = None
    polygon_buffer: float = 0.0
    image_list: list[str] | str | Path | None = None
    video_frames: int = 50
    ignore_folders: tuple[str, ...] = ()


def select_images(image_dir: str | Path, config: SfMConfig) -> list[Path]:
    """Apply the image selection options of ``config`` to a folder.

    Args:
        image_dir: Folder with input images.
        config: SfM options (``polygon``, ``image_stride``, ``max_images``).

    Returns:
        The selected image paths.

    Raises:
        ValueError: If the selection is empty.
    """
    from .selection import filter_by_polygon, subsample

    from .sources import scan_dataset

    root = Path(image_dir).resolve()
    images = scan_dataset(root, config.ignore_folders).photos
    if config.image_list is not None:
        names = config.image_list
        if isinstance(names, (str, Path)):
            names = [ln.strip() for ln in Path(names).read_text().splitlines() if ln.strip()]
        wanted = {str(n).replace("\\", "/").lower() for n in names}
        images = [p for p in images
                  if p.name.lower() in wanted or str(p.relative_to(root)).replace("\\", "/").lower() in wanted]
        logger.info("Image list: %d photos selected", len(images))
    if config.polygon:
        images = filter_by_polygon(images, config.polygon, config.polygon_buffer)
    images = subsample(images, config.max_images, config.image_stride)
    if len(images) < 2 and not scan_dataset(root, config.ignore_folders).videos:
        raise ValueError(f"Only {len(images)} image(s) selected; need at least 2.")
    return images


def list_images(image_dir: str | Path) -> list[Path]:
    """List image files in a directory (non-recursive), sorted by name.

    Args:
        image_dir: Directory to scan.

    Returns:
        Sorted list of image paths.

    Raises:
        FileNotFoundError: If the directory does not exist or contains no images.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images with extensions {sorted(IMAGE_EXTENSIONS)} in {image_dir}")
    return images


def prepared_name(src: Path) -> str:
    """File name a photo gets when prepared for SfM.

    Args:
        src: Original photo.

    Returns:
        Prepared file name (JPEG, or PNG for lossless inputs).
    """
    return src.stem + (".png" if src.suffix.lower() in {".png", ".tif", ".tiff", ".bmp"} else ".jpg")


def prepare_images(image_dir: str | Path | list[Path], out_dir: str | Path, max_image_size: int) -> Path:
    """Copy images to ``out_dir``, applying EXIF rotation and optional downscaling.

    Args:
        image_dir: Folder with the user's images, or an explicit list of image paths.
        out_dir: Destination folder.
        max_image_size: Maximum longest side in pixels (``0`` disables resizing).

    Returns:
        The destination folder (file names: :func:`prepared_name`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = image_dir if isinstance(image_dir, list) else list_images(image_dir)
    for src in sources:
        dst = out_dir / prepared_name(src)
        if dst.exists():
            continue
        with Image.open(src) as im:
            exif = im.getexif()
            exif.get_ifd(0x8825)  # load the GPS IFD so it is written back (COLMAP reads GPS priors)
            exif[0x0112] = 1  # orientation is applied below
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max_image_size and max(im.size) > max_image_size:
                scale = max_image_size / max(im.size)
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
            if dst.suffix == ".jpg":
                im.save(dst, quality=95, exif=exif.tobytes())
            else:
                im.save(dst)
    return out_dir


def _largest_reconstruction(recs: dict[int, pycolmap.Reconstruction]) -> pycolmap.Reconstruction:
    """Pick the reconstruction with the most registered images.

    Args:
        recs: Mapping returned by :func:`pycolmap.incremental_mapping`.

    Returns:
        The largest reconstruction.

    Raises:
        RuntimeError: If SfM produced no reconstruction.
    """
    if not recs:
        raise RuntimeError(
            "COLMAP could not reconstruct the scene. Make sure images overlap "
            "substantially (60-80%), are sharp, and show a static, textured scene."
        )
    return max(recs.values(), key=lambda r: r.num_reg_images())


class _DbProgress:
    """Progress bar fed by polling the COLMAP database while a stage runs in COLMAP.

    COLMAP's own log prints one block per image; geosplat silences it and shows
    one bar per stage instead.
    """

    def __init__(self, database: Path, query: str, total: int | None, desc: str) -> None:
        """Start polling.

        Args:
            database: COLMAP ``database.db``.
            query: SQL returning one count (e.g. images with keypoints).
            total: Expected final count (``None``: unknown).
            desc: Bar label.
        """
        import threading

        from tqdm.auto import tqdm

        self.database, self.query = database, query
        self.bar = tqdm(total=total, desc=desc, unit="img" if "keypoints" in query else "pair", leave=True)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def _count(self) -> int:
        """Current count in the database.

        Returns:
            Rows counted by the query (0 if the database is busy).
        """
        import sqlite3

        try:
            with sqlite3.connect(f"file:{self.database}?mode=ro", uri=True, timeout=0.2) as con:
                return int(con.execute(self.query).fetchone()[0])
        except sqlite3.Error:
            return self.bar.n

    def _poll(self) -> None:
        """Update the bar twice per second until stopped."""
        while not self.stop.wait(0.5):
            self.bar.update(max(0, self._count() - self.bar.n))

    def close(self) -> None:
        """Stop polling and finish the bar."""
        self.stop.set()
        self.thread.join()
        self.bar.update(max(0, self._count() - self.bar.n))
        self.bar.close()


def _quiet_colmap() -> None:
    """Show only COLMAP errors, unless geosplat logs at DEBUG level (``--verbose``)."""
    import pycolmap

    pycolmap.logging.minloglevel = 0 if logger.isEnabledFor(logging.DEBUG) else 2


def run_sfm(image_dir: str | Path, work_dir: str | Path, config: SfMConfig | None = None) -> Path:
    """Run the full COLMAP pipeline on a folder of images.

    Steps: image preparation, SIFT feature extraction, matching, incremental
    mapping and undistortion to a pinhole camera model.

    Args:
        image_dir: Folder with input images.
        work_dir: Folder for intermediate files (database, sparse models,
            undistorted images).
        config: SfM options. Defaults to :class:`SfMConfig`.

    Returns:
        Path to a COLMAP project folder (``images/`` + ``sparse/``) with
        undistorted pinhole cameras, ready for :func:`geosplat.scene.load_colmap_scene`.

    Raises:
        RuntimeError: If no reconstruction could be computed.
    """
    from .sources import extract_frames, group_slug, scan_dataset, _srt_for

    config = config or SfMConfig()
    work_dir = Path(work_dir)
    undistorted = work_dir / "undistorted"
    root = Path(image_dir).resolve()
    dataset = scan_dataset(root, config.ignore_folders)
    selected = select_images(root, config)
    videos = dataset.videos if config.video_frames > 0 else []
    signature = "\n".join([f"max_image_size={config.max_image_size}", f"video_frames={config.video_frames}"]
                          + [str(p.relative_to(root)) for p in selected] + [str(v.relative_to(root)) for v in videos])
    sig_file = work_dir / "selection.txt"
    same = sig_file.exists() and sig_file.read_text() == signature
    if (undistorted / "sparse").exists() and same and not config.overwrite:
        logger.info("Reusing existing SfM result in %s", undistorted)
        return undistorted
    if work_dir.exists() and (config.overwrite or not same):
        for sub in ("images", "database.db", "sparse", "undistorted", "catalogue.json"):
            p = work_dir / sub
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    images = work_dir / "images"
    flat = dataset.flat and not videos
    catalogue: dict[str, dict] = {}
    if flat:                                       # classic layout: one folder, one camera
        prepare_images(selected, images, config.max_image_size)
        for p in selected:
            catalogue[prepared_name(p)] = {"source": str(p), "kind": "photo", "group": "."}
    else:                                          # one subfolder (and one camera) per photo folder / video
        chosen_set = set(selected)
        for g in dataset.groups:
            chosen = [p for p in g.photos if p in chosen_set]
            if not chosen:
                continue
            slug = group_slug(g.label)
            prepare_images(chosen, images / slug, config.max_image_size)
            for p in chosen:
                catalogue[f"{slug}/{prepared_name(p)}"] = {"source": str(p), "kind": "photo", "group": g.label}
        for v in videos:
            slug = "video_" + group_slug(str(v.relative_to(root).with_suffix("")))
            logger.info("Extracting %d frames from %s", config.video_frames, v.name)
            srt = _srt_for(v, dataset.side_files)
            for frame, t in extract_frames(v, images / slug, config.video_frames, config.max_image_size):
                catalogue[f"{slug}/{frame.name}"] = {"source": str(v), "kind": "video_frame", "time": t,
                                                     "group": str(v.relative_to(root)), "srt": str(srt) if srt else None}
    (work_dir / "catalogue.json").write_text(json.dumps(
        {"root": str(root), "images": catalogue, "side_files": [str(p) for p in dataset.side_files]}, indent=1))
    sig_file.write_text(signature)
    n_images = len(catalogue)
    logger.info("SfM on %d images (%s)", n_images, dataset.summary())

    database = work_dir / "database.db"
    database.unlink(missing_ok=True)
    use_cuda = config.use_gpu and bool(getattr(pycolmap, "has_cuda", False))
    device = pycolmap.Device.cuda if use_cuda else pycolmap.Device.cpu
    logger.info("COLMAP device: %s", "cuda" if use_cuda else "cpu")

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = config.camera_model
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = max(config.max_image_size, 0) or 3200
    extraction.sift.max_num_features = config.max_num_features
    if not flat:
        camera_mode = pycolmap.CameraMode.PER_FOLDER   # each photo folder / video has its own camera
    else:
        camera_mode = pycolmap.CameraMode.SINGLE if config.single_camera else pycolmap.CameraMode.AUTO
    _quiet_colmap()
    bar = _DbProgress(database, "SELECT COUNT(*) FROM keypoints", n_images, "COLMAP features")
    try:
        pycolmap.extract_features(
            database, images, camera_mode=camera_mode, reader_options=reader,
            extraction_options=extraction, device=device,
        )
    finally:
        bar.close()

    matcher = config.matcher
    if matcher == "auto":
        from .geo import read_gps

        # Spatial matching only pairs images with GPS in their EXIF: use it only if all have it.
        with_gps = sum(e["kind"] == "photo" and read_gps(e["source"]) is not None for e in catalogue.values())
        matcher = "spatial" if with_gps == n_images and n_images >= 3 else "exhaustive"
    total = n_images * (n_images - 1) // 2 if matcher == "exhaustive" else None
    bar = _DbProgress(database, "SELECT COUNT(*) FROM two_view_geometries", total, f"COLMAP matching ({matcher})")
    try:
        _match(pycolmap, database, device, matcher, config)
    finally:
        bar.close()

    sparse = work_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    if config.mapper == "global":
        logger.info("COLMAP global mapping...")
        recs = pycolmap.global_mapping(database, images, sparse)
    elif config.mapper == "incremental":
        from tqdm.auto import tqdm

        options = pycolmap.IncrementalPipelineOptions()
        options.ba_use_gpu = use_cuda
        with tqdm(total=n_images, desc="COLMAP registered", unit="img") as reg:
            recs = pycolmap.incremental_mapping(database, images, sparse, options,
                                                initial_image_pair_callback=lambda: reg.update(2),
                                                next_image_callback=lambda: reg.update(1))
    else:
        raise ValueError(f"Unknown mapper {config.mapper!r}")
    rec = _largest_reconstruction(recs)
    logger.info("Registered %d/%d images, %d 3D points", rec.num_reg_images(), n_images, rec.num_points3D())
    best = sparse / "best"
    best.mkdir(exist_ok=True)
    rec.write(best)

    logger.info("Undistorting images...")
    pycolmap.undistort_images(undistorted, best, images)
    return undistorted


def _match(pycolmap: object, database: Path, device: object, matcher: str, config: SfMConfig) -> None:
    """Run the chosen COLMAP matcher.

    Args:
        pycolmap: The pycolmap module.
        database: COLMAP database.
        device: ``pycolmap.Device``.
        matcher: ``"spatial"``, ``"sequential"`` or ``"exhaustive"``.
        config: SfM options (neighbour counts).

    Raises:
        ValueError: For an unknown matcher.
    """
    if matcher == "spatial":
        pairing = pycolmap.SpatialPairingOptions()
        pairing.max_num_neighbors = config.spatial_neighbors
        pycolmap.match_spatial(database, pairing_options=pairing, device=device)
    elif matcher == "sequential":
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = config.sequential_overlap
        pycolmap.match_sequential(database, pairing_options=pairing, device=device)
    elif matcher == "exhaustive":
        pycolmap.match_exhaustive(database, device=device)
    else:
        raise ValueError(f"Unknown matcher {matcher!r}")
