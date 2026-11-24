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
        from .sources import dataset_positions

        # Prepared images of an SfM run: positions via the catalogue (photos and video frames
        # from any folder); a plain COLMAP project: the photos' own positions.
        positions = dataset_positions(Path(colmap_dir).parent, geo_config) or photo_positions(
            image_dir, geo_config.positions, geo_config.positions_crs, geo_config.gpx_time_offset)
        geo = georeference(raw, image_dir, geo_config.crs, positions)
    if geo is not None:
        down, north = gravity_in_scene(geo)
        scene = normalize_scene(raw, down=down, forward=north)
        geo = geo.compose(scene.transform)
    else:
        scene = normalize_scene(raw)
    return scene, geo


def find_picks(output_dir: str | Path, geo_config: GeoConfig) -> Path | None:
    """The picks file to apply: ``geo_config.located_targets`` or ``<output>/targets/located_targets.json``.

    Args:
        output_dir: Output folder of the run.
        geo_config: Georeferencing options.

    Returns:
        Path of the picks file, or ``None``.
    """
    if geo_config.located_targets:
        p = Path(geo_config.located_targets)
        if not p.is_file():
            raise FileNotFoundError(f"Picks file not found: {p}")
        return p
    p = Path(output_dir) / "targets" / "located_targets.json"
    return p if p.is_file() else None


def apply_picks(
    geo: GeoReference, scene: Scene, picks_file: Path, gcps: dict[str, np.ndarray], output_dir: Path
) -> GeoReference:
    """Correct the GPS georeference with targets located on an orthophoto.

    Picks from ``locate_targets.html`` carry the georeference of the orthophoto they
    were made on and their heights, so they stay valid when products are
    regenerated. Notebook picks (lat/lon only) refer to the GPS georeference and take
    heights from the DSM of the run.

    Args:
        geo: GPS-based georeference of ``scene``.
        scene: Normalized scene.
        picks_file: Saved picks.
        gcps: ``{name: [x, y, z]}`` in ``geo.crs``.
        output_dir: Output folder (DSM for picks without height, report).

    Returns:
        The corrected georeference, or ``geo`` if the picks do not apply.
    """
    from .gcp import load_picks, refine_georeference

    data = json.loads(picks_file.read_text())
    picks = load_picks(picks_file)
    ref = data.get("geo") if isinstance(data.get("geo"), dict) else None
    st = data.get("scene_transform")
    if st is not None and not np.allclose(np.asarray(st), scene.transform, atol=1e-6):
        logger.warning("%s was made on a different SfM reconstruction; ignoring it. Pick again on targets/locate_targets.html.",
                       picks_file)
        return geo
    base = geo
    if ref is not None:
        base = GeoReference(ref["crs"], np.asarray(ref["origin"]), np.asarray(ref["scene_to_geo"]),
                            ref.get("rmse_m", geo.rmse), ref.get("num_cameras", geo.num_cameras), dict(geo.extra))
        if base.crs != geo.crs:
            logger.warning("Picks were made in %s, the run is in %s; ignoring them.", base.crs, geo.crs)
            return geo
    dsm = next((p for p in (output_dir / "ortho_targets" / "dsm.tif", output_dir / "ortho" / "dsm.tif",
                            output_dir / "fast_orto" / "dsm_sparse.tif") if p.is_file()), None)
    try:
        new_geo, report = refine_georeference(base, picks, gcps, dsm)
    except ValueError as e:
        logger.warning("Picks not applied: %s", e)
        return geo
    (output_dir / "targets").mkdir(parents=True, exist_ok=True)
    (output_dir / "targets" / "report.json").write_text(json.dumps(report, indent=2))
    logger.info("Georeference corrected with %d picked targets (%s): RMSE xy %.3f m, z %.3f m",
                report["points"], report["method"], report["rmse_xy_m"], report["rmse_z_m"])
    return new_geo


def reuse_training(scene: Scene, output_dir: Path, config: TrainConfig) -> TrainResult | None:
    """Load the splat of a previous run if it was trained with the same scene and options.

    Args:
        scene: Normalized scene of this run.
        output_dir: Output folder (``splat.ply``, ``metrics.json``).
        config: Training options of this run.

    Returns:
        A :class:`TrainResult` built from the saved splat, or ``None`` if it must be retrained.
    """
    from dataclasses import asdict

    import torch

    from .gaussians import load_ply
    from .scene import split_train_test
    from .utils import resolve_backend

    ply, met = output_dir / "splat.ply", output_dir / "metrics.json"
    if not ply.is_file() or not met.is_file() or not torch.cuda.is_available():
        return None
    saved = json.loads(met.read_text())
    same_cfg = saved.get("config") == json.loads(json.dumps(asdict(config), default=str))
    same_scene = np.allclose(np.asarray(saved.get("colmap_to_scene_transform", np.eye(4))), scene.transform, atol=1e-6)
    if not (same_cfg and same_scene):
        return None
    device = torch.device("cuda")
    backend = resolve_backend(config.backend, device)
    params = load_ply(ply, device)
    sh_degree = int(round(np.sqrt(params["shN"].shape[1] + 1) - 1)) if "shN" in params else config.sh_degree
    train_cams, test_cams = split_train_test(scene.cameras, config.test_every)

    def at_res(cams: list) -> list:
        """Scale cameras to the training resolution.

        Args:
            cams: Full-resolution cameras.

        Returns:
            Scaled cameras.
        """
        r = config.max_resolution
        return [c.scaled(r / max(c.width, c.height)) if r and max(c.width, c.height) > r else c for c in cams]

    logger.info("Reusing the trained splat %s (same scene and training options)", ply)
    return TrainResult(params, ply, saved.get("metrics", {}), sh_degree, backend, at_res(train_cams), at_res(test_cams))


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
            ortho_targets/          # full-resolution orthophoto of geo_config.targets_orto (if set)
            targets/locate_targets.html         # locate the targets in a browser -> save targets/located_targets.json -> run again
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
    gcps = load_gcps(gcp_file, geo.crs, gcp_crs) if geo is not None and gcp_file is not None else None
    picks_file = find_picks(output_dir, geo_config) if geo is not None and gcps else None
    if picks_file is not None:
        gps_geo = geo
        geo = apply_picks(geo, scene, picks_file, gcps, output_dir)
        if geo is gps_geo:
            picks_file = None                                # not applied
    if geo is not None:
        geo.save(output_dir / "geo.json")
    # Without GPS, orthophotos are still made in a local Z-up frame (no CRS, SfM units).
    ortho_frame = geo if geo is not None else local_georeference(scene)
    targets_dir = None
    if geo_config.targets_orto and originals is not None:
        from .sources import resolve_subfolder

        targets_dir = resolve_subfolder(originals, geo_config.targets_orto)
    if geo_config.fast_orto:
        from .ortho import make_fast_orthophoto

        t = time.time()
        fast = make_fast_orthophoto(scene, ortho_frame, output_dir / "fast_orto", geo_config.gsd, gcps,
                                    max_view_angle=geo_config.ortho_max_angle,
                                    min_resolution=geo_config.ortho_min_resolution)
        outputs.update({k: v for k, v in fast.items() if isinstance(v, Path)})
        timings["fast_orto"] = time.time() - t
        logger.info("First orthophoto ready: %s", fast["fast_orto"])
    logger.info("Scene: %d cameras, %d points, scale %.3f", len(scene.cameras), len(scene.points), scene.scene_scale)

    t = time.time()
    train = reuse_training(scene, output_dir, train_config) or Trainer(scene, output_dir, train_config).train()
    timings["train"] = time.time() - t

    from .pointcloud import export_point_clouds

    t = time.time()
    outputs["splat"] = train.ply_path
    from .viewer import save_viewer

    from .pointcloud import dense_points_from_splat, processed_region

    outputs.update(export_point_clouds(train.params, scene, output_dir, geo, geo_config.dense_min_opacity))
    outputs["viewer"] = save_viewer(train.params, output_dir / "viewer.html", train.train_cameras,
                                    points=dense_points_from_splat(train.params, scene, geo_config.dense_min_opacity,
                                                                   return_index=True),
                                    region=processed_region(train.params, scene),
                                    m_per_unit=geo.scale if geo is not None else None)
    timings["point_clouds"] = time.time() - t

    splat_ok = train.backend == "gsplat"
    if targets_dir is not None:
        from .ortho import make_targets_orthophoto

        t = time.time()
        tg = make_targets_orthophoto(scene, ortho_frame, Path(colmap_dir).parent, targets_dir,
                                     output_dir / "ortho_targets", train.params if splat_ok else None,
                                     train.sh_degree, None, gcps, max_view_angle=geo_config.ortho_max_angle,
                                     min_resolution=geo_config.ortho_min_resolution)
        if tg:
            outputs["targets_orto"], outputs["targets_dsm"] = tg["targets_orto"], tg["targets_dsm"]
        timings["targets_orto"] = time.time() - t

    if geo_config.ortho:
        if not splat_ok:
            logger.warning("Orthophoto generation needs the gsplat CUDA backend; skipping.")
        else:
            from .ortho import make_orthophoto

            t = time.time()
            ortho = make_orthophoto(
                train.params, scene, ortho_frame, output_dir / "ortho", train.sh_degree, geo_config.gsd, gcps,
                max_view_angle=geo_config.ortho_max_angle, min_resolution=geo_config.ortho_min_resolution,
            )
            outputs.update({k: v for k, v in ortho.items() if isinstance(v, Path)})
            if "gcp_dz" in ortho:
                (output_dir / "ortho" / "gcp_check.json").write_text(json.dumps(ortho["gcp_dz"], indent=2))
            timings["orthophoto"] = time.time() - t

    if geo is not None and gcps:
        picker = write_picker(output_dir, outputs, gcps, geo, scene, picks_file)
        if picker is not None:
            outputs["locate_targets"] = picker

    timings["total"] = float(np.sum(list(timings.values())))
    (output_dir / "timings.json").write_text(json.dumps(timings, indent=2))
    logger.info("Timings (s): %s", {k: round(v, 1) for k, v in timings.items()})
    return PipelineResult(train, scene, geo, outputs, timings)


def write_picker(
    output_dir: Path, outputs: dict[str, Path], gcps: dict[str, np.ndarray], geo: GeoReference, scene: Scene,
    picks_file: Path | None,
) -> Path | None:
    """Write ``targets/locate_targets.html`` on the sharpest orthophoto of the run.

    Args:
        output_dir: Output folder.
        outputs: Products of the run (targets orthophoto, orthophoto or fast_orto).
        gcps: Ground control points in ``geo.crs``.
        geo: Georeference the products were made with.
        scene: Normalized scene (its transform is stored with the picks).
        picks_file: Picks applied in this run (shown again, moved to the corrected frame).

    Returns:
        Path of the page, or ``None`` without orthophoto.
    """
    from .picker_html import save_picker_html

    for o, d in (("targets_orto", "targets_dsm"), ("orthophoto", "dsm"), ("fast_orto", "dsm_sparse")):
        if o in outputs and d in outputs:
            ortho_tif, dsm_tif = outputs[o], outputs[d]
            break
    else:
        return None
    start = None
    report = output_dir / "targets" / "report.json"
    if picks_file is not None and report.is_file():
        from pyproj import Transformer

        to_ll = Transformer.from_crs(geo.crs, "EPSG:4326", always_xy=True)
        start = {}
        for n, (x, y, z) in json.loads(report.read_text()).get("picked_xyz_corrected", {}).items():
            lon, lat = to_ll.transform(x, y)
            start[n] = [lat, lon, z]
    html = save_picker_html(ortho_tif, dsm_tif, gcps, geo, output_dir / "targets" / "locate_targets.html", start, scene.transform)
    logger.info("Locate the targets in %s, save located_targets.json next to it and run again to correct the georeference.",
                html)
    return html
