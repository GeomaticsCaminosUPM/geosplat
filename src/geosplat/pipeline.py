"""End-to-end pipeline: folder of images -> point clouds, Gaussian splat, orthophoto."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geo import GeoConfig, GeoReference, gcp_setup, georeference, gravity_in_scene, load_gcps, local_georeference
from .scene import Scene, is_colmap_project, load_colmap_scene, normalize_scene
from .sfm import SfMConfig, run_sfm
from .trainer import TrainConfig, Trainer, TrainResult

logger = logging.getLogger("geosplat")


@dataclass
class PipelineResult:
    """Everything produced by :func:`images_to_splat`.

    Attributes:
        train: Training result (model parameters, metrics, cameras).
        scene: The (normalized) scene used for training.
        geo: Georeference, if the images had GPS.
        outputs: Paths of the written products (``splat``, ``points_sparse``,
            ``points_dense``, ``orthophoto``, ``orthophoto_splat``, ``dsm``...).
        timings: Wall-clock seconds per stage.
    """

    train: TrainResult
    scene: Scene
    geo: GeoReference | None
    outputs: dict[str, Path] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def ply_path(self) -> Path:
        """Path of the Gaussian splat ``.ply``.

        Returns:
            ``output_dir / "splat.ply"``.
        """
        return self.train.ply_path

    @property
    def metrics(self) -> dict[str, float | str]:
        """Evaluation metrics of the training stage.

        Returns:
            PSNR / SSIM / number of Gaussians / training time.
        """
        return self.train.metrics


def prepare_scene(
    colmap_dir: str | Path, image_dir: str | Path | None, geo_config: GeoConfig | None
) -> tuple[Scene, GeoReference | None]:
    """Load a COLMAP project, georeference it if possible and normalize it.

    Args:
        colmap_dir: Undistorted COLMAP project.
        image_dir: Folder with the original (EXIF/GPS-tagged) images, or ``None``.
        geo_config: Georeferencing options (``None`` disables georeferencing).

    Returns:
        Tuple ``(scene, geo)``. The scene is centered and upright (gravity from
        GPS when available); ``geo`` maps scene coordinates to the CRS.
    """
    raw = load_colmap_scene(colmap_dir, normalize=False)
    geo = None
    if geo_config is not None and geo_config.enabled and image_dir is not None:
        from .positions import photo_positions

        positions = photo_positions(image_dir, geo_config.positions, geo_config.positions_crs,
                                    geo_config.gpx_time_offset)
        geo = georeference(raw, image_dir, geo_config.crs, positions)
    if geo is not None:
        down, north = gravity_in_scene(geo)
        scene = normalize_scene(raw, down=down, forward=north)
        geo = geo.compose(scene.transform)
    else:
        scene = normalize_scene(raw)
    return scene, geo


def images_to_splat(
    image_dir: str | Path,
    output_dir: str | Path,
    sfm_config: SfMConfig | None = None,
    train_config: TrainConfig | None = None,
    geo_config: GeoConfig | None = None,
    preset: str | None = None,
) -> PipelineResult:
    """Build point clouds, a Gaussian splat and (for GPS imagery) an orthophoto.

    If ``image_dir`` is already an undistorted COLMAP project (``images/`` +
    ``sparse/``) the SfM stage is skipped.

    Output layout::

        output_dir/
            colmap/                 # SfM work dir
            splat.ply               # Gaussian splat (3DGS format) -> SuperSplat
            viewer.html             # interactive splat viewer (open in a browser)
            points_sparse.ply       # SfM points (CRS coordinates if georeferenced)
            points_dense.ply        # dense colored points from the splat
            geo.json                # georeference (if images have GPS)
            fast_orto/fast_orto.tif # first orthophoto right after SfM (sparse DSM)
            ortho/orthophoto.tif    # true orthophoto from the photos
                                    # (CRS-georeferenced with GPS; local frame, no CRS, otherwise)
            ortho/orthophoto_splat.tif, ortho/dsm.tif, ortho/orthophoto_preview.jpg
            metrics.json, renders/, timings.json

    Args:
        image_dir: Folder with the input images, a ``.zip`` of them (extracted
            automatically), or an undistorted COLMAP project.
        output_dir: Folder for all outputs.
        sfm_config: Structure-from-Motion and image-selection options.
        train_config: Training options.
        geo_config: Georeferencing / orthophoto options (default: automatic
            when images carry GPS).
        preset: Speed preset (``"very_fast"``, ``"fast"``, ``"medium"``,
            ``"quality"``) used for any config not passed explicitly. ``None``
            means ``"medium"``. To tweak a preset, build configs with
            :func:`geosplat.presets.make_configs`.

    Returns:
        A :class:`PipelineResult`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from .presets import DEFAULT_PRESET, make_configs

    p_sfm, p_train, p_geo = make_configs(preset or DEFAULT_PRESET)
    sfm_config = sfm_config or p_sfm
    train_config = train_config or p_train
    geo_config = geo_config or p_geo
    timings: dict[str, float] = {}

    t = time.time()
    from .datasets import resolve_image_source

    image_dir = resolve_image_source(image_dir)          # folder, dataset folder or .zip of photos
    if is_colmap_project(image_dir):
        logger.info("%s is a COLMAP project; skipping SfM", image_dir)
        colmap_dir, originals = Path(image_dir), None
    else:
        colmap_dir, originals = run_sfm(image_dir, output_dir / "colmap", sfm_config), image_dir
    timings["sfm"] = time.time() - t

    # GCP file and CRS first: a projected GCP CRS becomes the output CRS if none is set.
    gcp_file, gcp_crs = gcp_setup(originals, geo_config) if originals is not None and geo_config.enabled else (None, None)
    scene, geo = prepare_scene(colmap_dir, originals, geo_config)
    outputs: dict[str, Path] = {}
    if geo is not None:
        geo.save(output_dir / "geo.json")
    # Without GPS, orthophotos are still made in a local Z-up frame (no CRS, SfM units).
    ortho_frame = geo if geo is not None else local_georeference(scene)
    gcps = load_gcps(gcp_file, geo.crs, gcp_crs) if geo is not None and gcp_file is not None else None
    if geo_config.fast_orto:
        from .ortho import make_fast_orthophoto

        t = time.time()
        fast = make_fast_orthophoto(scene, ortho_frame, output_dir / "fast_orto", geo_config.gsd,
                                    gcps)
        outputs.update({k: v for k, v in fast.items() if isinstance(v, Path)})
        timings["fast_orto"] = time.time() - t
        logger.info("First orthophoto ready: %s", fast["fast_orto"])
    logger.info("Scene: %d cameras, %d points, scale %.3f", len(scene.cameras), len(scene.points), scene.scene_scale)

    t = time.time()
    train = Trainer(scene, output_dir, train_config).train()
    timings["train"] = time.time() - t

    from .pointcloud import export_point_clouds

    t = time.time()
    outputs["splat"] = train.ply_path
    from .viewer import save_viewer

    outputs["viewer"] = save_viewer(train.params, output_dir / "viewer.html", train.train_cameras)
    outputs.update(export_point_clouds(train.params, scene, output_dir, geo, geo_config.dense_min_opacity))
    timings["point_clouds"] = time.time() - t

    if geo_config.ortho:
        if train.backend != "gsplat":
            logger.warning("Orthophoto generation needs the gsplat CUDA backend; skipping.")
        else:
            from .ortho import make_orthophoto

            t = time.time()
            ortho = make_orthophoto(
                train.params, scene, ortho_frame, output_dir / "ortho", train.sh_degree, geo_config.gsd, gcps,
            )
            outputs.update({k: v for k, v in ortho.items() if isinstance(v, Path)})
            if "gcp_dz" in ortho:
                (output_dir / "ortho" / "gcp_check.json").write_text(json.dumps(ortho["gcp_dz"], indent=2))
            timings["orthophoto"] = time.time() - t

    timings["total"] = float(np.sum(list(timings.values())))
    (output_dir / "timings.json").write_text(json.dumps(timings, indent=2))
    logger.info("Timings (s): %s", {k: round(v, 1) for k, v in timings.items()})
    return PipelineResult(train, scene, geo, outputs, timings)
