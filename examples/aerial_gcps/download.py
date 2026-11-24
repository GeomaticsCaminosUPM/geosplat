"""Download a block of Agisoft's *Aerial images with GCPs* sample into ``examples/aerial_gcps/images``.

Survey flight (Sony RX1R, data by GeoScan) with a camera positions table and
surveyed ground control points. Only the needed files are read from the 15 GB
archive: the photos nearest to GCPs 4, 5 and 9, ``Cameras_WGS84.txt`` and
``GCPs_WGS84.txt`` (the GCPs inside the block).

Usage::

    uv run python examples/aerial_gcps/download.py                    # 40 photos (~440 MB)
    uv run python examples/aerial_gcps/download.py --num-images 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

from geosplat.datasets import download_aerial_gcps


def main() -> None:
    """Parse arguments and download the photos next to this script."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-images", type=int, default=40, help="Number of photos (nearest to the GCP block).")
    args = p.parse_args()
    out = download_aerial_gcps(Path(__file__).resolve().parent / "images", num_images=args.num_images)
    print(f"Photos, positions and GCPs in {out}")


if __name__ == "__main__":
    main()
