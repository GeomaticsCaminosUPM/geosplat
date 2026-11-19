"""Run geosplat on any photo folder: fast orthophoto, point clouds, Gaussian splat, true orthophoto.

``SOURCE`` is either the name of a folder in ``examples/`` (``dron2024``) or any
path (``/data/flight1`` or ``/data/flight1/images``). Photos are read from
``<dataset>/images/`` if it exists, otherwise from the folder itself. Results go
to ``<dataset>/output_<preset>/`` unless ``-o`` is given.

GPS photos need no configuration: positions and the UTM zone come from the photo
metadata, and a ``name,x,y,z`` CSV next to the photos is used as ground control
points. Every option of the ``geosplat`` command is available here; an optional
JSON file (``--config``, or ``<dataset>/config.json``) can hold defaults::

    {"preset": "fast", "sfm": {"max_images": 60}, "train": {"iterations": 4000}, "geo": {"crs": "EPSG:25830"}}

Priority: command line > config file > preset > defaults.

Examples::

    uv run python examples/run.py dron2024                          # medium preset
    uv run python examples/run.py dron2024 --preset very_fast
    uv run python examples/run.py /data/flight1 --preset fast --max-images 80 --polygon aoi.gpkg
    uv run python examples/run.py my_scene --preset quality --crs EPSG:25830 -o /data/results
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from geosplat import images_to_splat, render_path, save_gif
from geosplat.cli import add_options, configs_from_args

EXAMPLES_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="Example name (folder in examples/) or path to a photo folder.")
    p.add_argument("-o", "--output", help="Output folder (default: <dataset>/output_<preset>).")
    p.add_argument("--config", help="JSON file with preset/sfm/train/geo defaults (default: <dataset>/config.json).")
    p.add_argument("--video-frames", type=int, default=90, help="Frames of the fly-through GIF (0 to skip).")
    return add_options(p).parse_args()


def resolve_source(source: str) -> tuple[Path, Path]:
    """Find the dataset folder and the photo folder for ``source``.

    Args:
        source: Example name or path.

    Returns:
        Tuple ``(dataset_dir, images_dir)``.

    Raises:
        SystemExit: If no photo folder is found.
    """
    path = Path(source).expanduser()
    if not path.is_dir():
        path = EXAMPLES_DIR / source
    if (path / "images").is_dir():
        return path, path / "images"
    if path.is_dir():
        return (path.parent if path.name == "images" else path), path
    hint = f" (run examples/{source}/download.py first)" if (EXAMPLES_DIR / source / "download.py").exists() else ""
    raise SystemExit(f"No photo folder found for {source!r}{hint}")


def load_config_file(dataset_dir: Path, config: str | None) -> dict:
    """Load the optional JSON config, resolving relative paths against its folder.

    Args:
        dataset_dir: Dataset folder (used for the default ``config.json``).
        config: Explicit config path, or ``None``.

    Returns:
        Config dict (empty if there is no file).
    """
    path = Path(config) if config else dataset_dir / "config.json"
    if not path.is_file():
        return {}
    cfg = json.loads(path.read_text())
    for section, key in (("sfm", "polygon"), ("geo", "gcps")):
        value = cfg.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            cfg[section][key] = str((path.parent / value).resolve())
    print(f"Config file: {path}")
    return cfg


def main() -> None:
    """Resolve paths and options, run the pipeline and print the outputs."""
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")
    dataset_dir, images_dir = resolve_source(args.source)
    preset, sfm, train, geo = configs_from_args(args, load_config_file(dataset_dir, args.config))
    out_dir = Path(args.output) if args.output else dataset_dir / f"output_{preset}"
    print(f"Photos: {images_dir}\nOutput: {out_dir}\nPreset: {preset} "
          f"({train.iterations} steps, {train.max_resolution} px, SfM {sfm.max_image_size} px)")

    result = images_to_splat(images_dir, out_dir, sfm, train, geo)
    if args.video_frames:
        frames = render_path(result.train.params, result.train.train_cameras, result.train.sh_degree,
                             result.train.backend, args.video_frames)
        result.outputs["flythrough"] = save_gif(frames, out_dir / "flythrough.gif", fps=20, max_width=480)

    print()
    for name, path in result.outputs.items():
        print(f"{name:>18}: {path}")
    print(f"{'metrics':>18}: {result.metrics}")
    print(f"{'timings (s)':>18}: { {k: round(v, 1) for k, v in result.timings.items()} }")
    print("Open viewer.html in a browser, splat.ply in https://superspl.at/editor, the .tif files in QGIS.")


if __name__ == "__main__":
    main()
