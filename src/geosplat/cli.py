"""Command-line interface: ``geosplat IMAGE_DIR -o OUTPUT_DIR``."""

from __future__ import annotations

import argparse
import logging

from .geo import GeoConfig
from .pipeline import images_to_splat
from .presets import DEFAULT_PRESET, PRESETS, make_configs
from .sfm import SfMConfig
from .trainer import TrainConfig


def add_options(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add every pipeline option (preset, selection, SfM, training, geo) to a parser.

    All options default to ``None`` so that :func:`configs_from_args` can tell
    "not given" (keep preset / config file) from an explicit value.

    Args:
        p: Parser to extend.

    Returns:
        The same parser.
    """
    p.add_argument(
        "-p", "--preset", default=None, choices=sorted(PRESETS) + ["very-fast"],
        help="Speed preset: very_fast (~1 min), fast (~2 min), medium (~13 min, default), quality (~1 h). "
             "Times for 40 drone photos on an RTX 4060; explicit options override the preset.",
    )

    sel = p.add_argument_group("image selection (speed)")
    sel.add_argument("--max-images", type=int, default=None, help="Use at most N evenly spaced images.")
    sel.add_argument("--image-stride", type=int, default=None, help="Use every N-th image.")
    sel.add_argument("--polygon", default=None, help="Area of interest (.gpkg/.shp/.geojson); GPS images inside only.")
    sel.add_argument("--polygon-buffer", type=float, default=None, help="Grow the polygon by this many meters.")
    sel.add_argument("--max-image-size", type=int, default=None, help="Downscale images (px).")

    sfm = p.add_argument_group("structure from motion")
    sfm.add_argument("--matcher", choices=["auto", "exhaustive", "sequential", "spatial"], default=None,
                     help="auto = spatial (GPS) if available, else exhaustive.")
    sfm.add_argument("--mapper", choices=["incremental", "global"], default=None,
                     help="global (GLOMAP) is faster for many images.")
    sfm.add_argument("--camera-model", default=None, help="COLMAP camera model (default SIMPLE_RADIAL).")
    sfm.add_argument("--max-features", type=int, default=None, help="Max SIFT features per image.")
    sfm.add_argument("--multi-camera", action="store_true", help="Images come from different cameras / zoom levels.")
    sfm.add_argument("--overwrite-sfm", action="store_true", help="Re-run SfM even if results exist.")

    tr = p.add_argument_group("training")
    tr.add_argument("--iterations", type=int, default=None, help="Training steps.")
    tr.add_argument("--max-resolution", type=int, default=None, help="Longest training side (px).")
    tr.add_argument("--sh-degree", type=int, default=None, help="Spherical harmonics degree (0-3).")
    tr.add_argument("--max-gaussians", type=int, default=None, help="Cap the number of Gaussians (MCMC strategy).")
    tr.add_argument("--early-stop", action="store_true", default=None, help="Stop when held-out PSNR stops improving.")
    tr.add_argument("--no-progressive", action="store_true", help="Disable coarse-to-fine resolution schedule.")
    tr.add_argument("--no-selective-adam", action="store_true", help="Use plain Adam instead of SelectiveAdam.")
    tr.add_argument("--no-background-sphere", action="store_true", help="Do not add far background Gaussians.")
    tr.add_argument("--test-every", type=int, default=None, help="Hold out every N-th image (0 = none).")
    tr.add_argument("--seed", type=int, default=None, help="Random seed.")
    tr.add_argument("--backend", choices=["auto", "gsplat", "torch"], default=None, help="Rasterizer backend.")
    tr.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu (default: auto).")

    geo = p.add_argument_group("georeferencing / orthophoto")
    geo.add_argument("--crs", default=None, help="Output CRS, e.g. EPSG:25830 (default: UTM zone from photo GPS).")
    geo.add_argument("--gsd", type=float, default=None, help="Orthophoto pixel size (default: native).")
    geo.add_argument("--gcps", default=None, help="CSV name,x,y,z in the output CRS (default: a CSV next to the photos).")
    geo.add_argument("--dense-min-opacity", default=None,
                     type=lambda v: v if v == "auto" else float(v),
                     help="Min Gaussian opacity (0-1) for points_dense.ply, or 'auto' (default, Otsu).")
    geo.add_argument("--no-geo", action="store_true", help="Do not georeference (ignore photo GPS).")
    geo.add_argument("--no-ortho", action="store_true", help="Do not produce the final orthophoto / DSM.")
    geo.add_argument("--no-fast-orto", action="store_true", help="Do not produce the fast orthophoto.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p


def configs_from_args(
    a: argparse.Namespace, base: dict | None = None
) -> tuple[str, SfMConfig, TrainConfig, GeoConfig]:
    """Merge preset, an optional config dict and command-line options.

    Priority: command line > ``base`` (e.g. a ``config.json``) > preset > defaults.

    Args:
        a: Parsed arguments (from a parser built with :func:`add_options`).
        base: Optional dict with ``"preset"``, ``"sfm"``, ``"train"`` and ``"geo"`` keys.

    Returns:
        Tuple ``(preset, sfm_config, train_config, geo_config)``.
    """
    base = base or {}
    preset = (a.preset or base.get("preset") or DEFAULT_PRESET).replace("-", "_")
    off = lambda flag: False if flag else None  # noqa: E731 - "--no-x" given -> False, else keep
    on = lambda flag: True if flag else None  # noqa: E731
    cli_sfm = dict(
        max_image_size=a.max_image_size, matcher=a.matcher, mapper=a.mapper, camera_model=a.camera_model,
        max_num_features=a.max_features, single_camera=off(a.multi_camera), overwrite=on(a.overwrite_sfm),
        max_images=a.max_images, image_stride=a.image_stride, polygon=a.polygon, polygon_buffer=a.polygon_buffer,
    )
    cli_train = dict(
        iterations=a.iterations, max_resolution=a.max_resolution, sh_degree=a.sh_degree,
        max_gaussians=a.max_gaussians, test_every=a.test_every, backend=a.backend, device=a.device,
        early_stop=a.early_stop, progressive_resolution=off(a.no_progressive),
        selective_adam=off(a.no_selective_adam), seed=a.seed,
        background_points=0 if a.no_background_sphere else None,
    )
    cli_geo = dict(
        enabled=off(a.no_geo), crs=a.crs, ortho=off(a.no_ortho), fast_orto=off(a.no_fast_orto), gsd=a.gsd, gcps=a.gcps,
        dense_min_opacity=a.dense_min_opacity,
    )

    def merged(section: str, cli: dict) -> dict:
        """Overlay command-line values on the config-file section.

        Args:
            section: ``"sfm"``, ``"train"`` or ``"geo"``.
            cli: Command-line values (``None`` = not given).

        Returns:
            Merged overrides.
        """
        return {**base.get(section, {}), **{k: v for k, v in cli.items() if v is not None}}

    sfm, train, geo = make_configs(preset, merged("sfm", cli_sfm), merged("train", cli_train), merged("geo", cli_geo))
    return preset, sfm, train, geo


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser of the ``geosplat`` command.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(
        prog="geosplat",
        description="Fast orthophoto, point clouds, Gaussian splat and true orthophoto from a folder of photos.",
    )
    p.add_argument("image_dir", help="Folder with input images (or an undistorted COLMAP project).")
    p.add_argument("-o", "--output", required=True, help="Output folder.")
    return add_options(p)


def main(argv: list[str] | None = None) -> None:
    """Entry point of the ``geosplat`` command.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(message)s")
    preset, sfm, train, geo = configs_from_args(a)
    logging.info("Preset %s: %d steps at %d px, SfM images %d px", preset, train.iterations,
                 train.max_resolution, sfm.max_image_size)
    result = images_to_splat(a.image_dir, a.output, sfm, train, geo)
    for name, path in result.outputs.items():
        print(f"{name:>18}: {path}")
    print("metrics:", result.metrics)
    print("timings (s):", {k: round(v, 1) for k, v in result.timings.items()})
