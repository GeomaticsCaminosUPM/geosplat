"""Choosing which images to process (subsampling, polygon filtering)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("geosplat")


def subsample(images: list[Path], max_images: int | None = None, stride: int = 1) -> list[Path]:
    """Reduce an ordered image list.

    Args:
        images: Images sorted by name (capture order for most cameras).
        max_images: Keep at most this many, evenly spaced over the sequence.
        stride: First keep every ``stride``-th image.

    Returns:
        The selected images (order preserved).
    """
    images = images[:: max(stride, 1)]
    if max_images and len(images) > max_images:
        idx = np.unique(np.linspace(0, len(images) - 1, max_images).round().astype(int))
        images = [images[i] for i in idx]
    return images


def read_polygon(path: str | Path) -> tuple[object, str | None]:
    """Read and union all polygons of a vector file (GeoPackage, Shapefile, GeoJSON...).

    Args:
        path: Vector file readable by GDAL/OGR (``.gpkg``, ``.shp``, ``.geojson``...).

    Returns:
        Tuple ``(shapely geometry, crs)`` where ``crs`` is a string understood
        by pyproj (``None`` if the file has no CRS; WGS84 is then assumed).

    Raises:
        ValueError: If the file has no geometries.
    """
    import pyogrio
    import shapely

    meta, _, geoms, _ = pyogrio.raw.read(str(path))
    shapes = [g for g in shapely.from_wkb(geoms) if g is not None]
    if not shapes:
        raise ValueError(f"No geometries in {path}")
    return shapely.union_all(shapes), meta.get("crs")


def filter_by_polygon(images: list[Path], polygon_path: str | Path, buffer_m: float = 0.0) -> list[Path]:
    """Keep images whose GPS position lies inside a polygon.

    Args:
        images: Candidate images (must carry GPS EXIF tags).
        polygon_path: Vector file with the area of interest.
        buffer_m: Grow the polygon by this distance in meters (negative shrinks).
            Useful to include photos just outside the area that still see it.

    Returns:
        The images inside the (buffered) polygon. Images without GPS are dropped.
    """
    import shapely
    from pyproj import CRS, Transformer
    from shapely.ops import transform as shapely_transform

    from .geo import read_gps, utm_crs_for

    poly, crs = read_polygon(polygon_path)
    crs = CRS.from_user_input(crs) if crs else CRS.from_epsg(4326)
    gps = {p: read_gps(p) for p in images}
    with_gps = [p for p, g in gps.items() if g is not None]
    if not with_gps:
        raise ValueError("Polygon filtering needs GPS-tagged images, but none have GPS EXIF data.")
    lat = np.array([gps[p][0] for p in with_gps])
    lon = np.array([gps[p][1] for p in with_gps])

    if buffer_m and not crs.is_projected:
        # Buffer in meters: work in the local UTM zone.
        utm = CRS.from_user_input(utm_crs_for(float(np.median(lon)), float(np.median(lat))))
        poly = shapely_transform(Transformer.from_crs(crs, utm, always_xy=True).transform, poly)
        crs = utm
    if buffer_m:
        poly = poly.buffer(buffer_m)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    inside = shapely.contains_xy(poly, x, y)
    selected = [p for p, ok in zip(with_gps, inside) if ok]
    logger.info("Polygon filter: %d of %d images inside %s", len(selected), len(images), Path(polygon_path).name)
    return selected
