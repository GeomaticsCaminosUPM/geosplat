"""Download OpenDroneMap's *Seneca* drone photos (GPS-tagged) into ``examples/seneca/images``.

Usage::

    uv run python examples/seneca/download.py            # 40 photos (~90 MB)
    uv run python examples/seneca/download.py --all      # all 167 photos (~380 MB)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from geosplat.datasets import download_seneca


def main() -> None:
    """Parse arguments and download the photos next to this script."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-images", type=int, default=40, help="Number of photos to keep (flight order).")
    p.add_argument("--all", action="store_true", help="Keep all 167 photos.")
    args = p.parse_args()
    out = download_seneca(Path(__file__).resolve().parent / "images", num_images=None if args.all else args.num_images)
    print(f"Photos in {out}")


if __name__ == "__main__":
    main()
