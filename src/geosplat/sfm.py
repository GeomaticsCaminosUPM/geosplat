"""Structure-from-Motion with pycolmap: images -> camera poses + sparse points."""

from __future__ import annotations

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
        image_list: Only use these photos: a list of file names, or a text file
            with one name per line (e.g. saved by the notebook photo map).
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

    images = list_images(image_dir)
    if config.image_list is not None:
        names = config.image_list
        if isinstance(names, (str, Path)):
            names = Path(names).read_text().split()
        wanted = {Path(n).name for n in names}
        images = [p for p in images if p.name in wanted]
        logger.info("Image list: %d photos selected", len(images))
    if config.polygon:
        images = filter_by_polygon(images, config.polygon, config.polygon_buffer)
    images = subsample(images, config.max_images, config.image_stride)
    if len(images) < 2:
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


def prepare_images(image_dir: str | Path | list[Path], out_dir: str | Path, max_image_size: int) -> Path:
    """Copy images to ``out_dir``, applying EXIF rotation and optional downscaling.

    Args:
        image_dir: Folder with the user's images, or an explicit list of image paths.
        out_dir: Destination folder.
        max_image_size: Maximum longest side in pixels (``0`` disables resizing).

    Returns:
        The destination folder.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = image_dir if isinstance(image_dir, list) else list_images(image_dir)
    for src in sources:
        dst = out_dir / (src.stem + (".png" if src.suffix.lower() in {".png", ".tif", ".tiff", ".bmp"} else ".jpg"))
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
    config = config or SfMConfig()
    work_dir = Path(work_dir)
    undistorted = work_dir / "undistorted"
    selected = select_images(image_dir, config)
    signature = "\n".join([f"max_image_size={config.max_image_size}"] + [p.name for p in selected])
    sig_file = work_dir / "selection.txt"
    same = sig_file.exists() and sig_file.read_text() == signature
    if (undistorted / "sparse").exists() and same and not config.overwrite:
        logger.info("Reusing existing SfM result in %s", undistorted)
        return undistorted
    if work_dir.exists() and (config.overwrite or not same):
        for sub in ("images", "database.db", "sparse", "undistorted"):
            p = work_dir / sub
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    images = prepare_images(selected, work_dir / "images", config.max_image_size)
    sig_file.write_text(signature)
    n_images = len(list_images(images))
    logger.info("SfM on %d images", n_images)

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
    camera_mode = pycolmap.CameraMode.SINGLE if config.single_camera else pycolmap.CameraMode.AUTO
    logger.info("Extracting features...")
    pycolmap.extract_features(
        database, images, camera_mode=camera_mode, reader_options=reader,
        extraction_options=extraction, device=device,
    )

    matcher = config.matcher
    if matcher == "auto":
        from .geo import read_gps

        matcher = "spatial" if sum(read_gps(p) is not None for p in selected) >= 3 else "exhaustive"
    logger.info("Matching features (%s)...", matcher)
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

    logger.info("Running %s mapping...", config.mapper)
    sparse = work_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    if config.mapper == "global":
        recs = pycolmap.global_mapping(database, images, sparse)
    elif config.mapper == "incremental":
        options = pycolmap.IncrementalPipelineOptions()
        options.ba_use_gpu = use_cuda
        recs = pycolmap.incremental_mapping(database, images, sparse, options)
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
