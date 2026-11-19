"""Download the COLMAP *South Building* photos into ``examples/south_building/images``.

Usage::

    uv run python examples/south_building/download.py            # 40 images (fast)
    uv run python examples/south_building/download.py --all      # all 128 images
"""

from __future__ import annotations

import argparse
from pathlib import Path

from geosplat.datasets import download_south_building


def main() -> None:
    """Parse arguments and download the images next to this script."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-images", type=int, default=40, help="Number of images to keep.")
    p.add_argument("--all", action="store_true", help="Keep all 128 images.")
    args = p.parse_args()
    out = download_south_building(
        Path(__file__).resolve().parent / "images", num_images=None if args.all else args.num_images
    )
    print(f"Images in {out}")


if __name__ == "__main__":
    main()
