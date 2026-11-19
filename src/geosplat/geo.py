"""Georeferencing from image GPS (EXIF): COLMAP frame -> projected CRS (e.g. UTM)."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image

from .scene import Scene
from .sfm import list_images

logger = logging.getLogger("geosplat")


@dataclass
class GeoConfig:
    """Options for georeferencing and output products (orthophotos, DSM, point clouds).

    Attributes:
        enabled: Georeference automatically when images carry GPS EXIF tags.
        crs: Target projected CRS (e.g. ``"EPSG:25830"`` for ETRS89 / UTM 30N).
            ``None`` picks the WGS84 UTM zone of the images.
        ortho: Render a true orthophoto and a DSM from the trained splat.
        fast_orto: Right after SfM (before training), produce a first
            orthophoto from the photos on a DSM interpolated from the sparse
            points (``fast_orto/fast_orto.tif``). Takes seconds.
        gsd: Orthophoto ground sampling distance in meters per pixel. ``None``
            estimates it from flight height and camera focal length.
        gcps: Optional CSV of ground control points (``name,x,y,z`` in ``crs``)
            drawn on the orthophoto preview to check the georeferencing.
            ``None`` uses a ``*.csv`` found in the photo folder, if any.
        dense_min_opacity: Minimum Gaussian opacity (0-1) for a Gaussian center
            to become a point of ``points_dense.ply``, or ``"auto"`` (default)
            to choose it with Otsu's method on the opacity histogram. Lower =
            more (noisier) points; higher = fewer, cleaner points.
    """

    enabled: bool = True
    crs: str | None = None
    ortho: bool = True
    fast_orto: bool = True
    dense_min_opacity: float | str = "auto"
    gsd: float | None = None
    gcps: str | Path | None = None


@dataclass
class GeoReference:
    """Similarity transform from scene coordinates to a projected CRS.

    Coordinates are stored relative to ``origin`` to keep float32 precision:
    ``crs_coords = origin + (scene_to_geo @ [x, y, z, 1])[:3]``. The local geo
    frame is X = easting, Y = northing, Z = height (meters).

    Attributes:
        crs: CRS identifier understood by pyproj (e.g. ``"EPSG:32630"``), or
            ``None`` for a local, non-georeferenced frame.
        origin: Offset ``[3]`` (easting, northing, height) in the CRS.
        scene_to_geo: ``[4, 4]`` similarity matrix (scale * rotation, translation).
        rmse: Root-mean-square distance in meters between SfM camera centers
            (after alignment) and their GPS positions.
        num_cameras: Number of cameras used in the fit.
    """

    crs: str | None
    origin: np.ndarray
    scene_to_geo: np.ndarray
    rmse: float
    num_cameras: int
    extra: dict = field(default_factory=dict)

    @property
    def scale(self) -> float:
        """Meters per scene unit.

        Returns:
            Uniform scale of the similarity transform.
        """
        return float(np.cbrt(np.linalg.det(self.scene_to_geo[:3, :3])))

    @property
    def rotation(self) -> np.ndarray:
        """Rotation part of the similarity transform.

        Returns:
            ``[3, 3]`` rotation matrix.
        """
        return self.scene_to_geo[:3, :3] / self.scale

    def scene_to_local(self, pts: np.ndarray) -> np.ndarray:
        """Map scene points to local geo coordinates (CRS minus ``origin``).

        Args:
            pts: Points ``[N, 3]`` in scene coordinates.

        Returns:
            Points ``[N, 3]`` in local geo coordinates (meters).
        """
        return pts @ self.scene_to_geo[:3, :3].T + self.scene_to_geo[:3, 3]

    def compose(self, transform: np.ndarray) -> "GeoReference":
        """Re-express the georeference for a scene transformed by ``transform``.

        Args:
            transform: ``[4, 4]`` matrix mapping current scene coordinates to the
                new scene coordinates.

        Returns:
            A new :class:`GeoReference` for the new scene coordinates.
        """
        return GeoReference(
            self.crs, self.origin.copy(), self.scene_to_geo @ np.linalg.inv(transform),
            self.rmse, self.num_cameras, dict(self.extra),
        )

    def save(self, path: str | Path) -> None:
        """Write the georeference to JSON.

        Args:
            path: Output ``.json`` path.
        """
        data = {
            "crs": self.crs,
            "origin": self.origin.tolist(),
            "scene_to_geo": self.scene_to_geo.tolist(),
            "scale_m_per_unit": self.scale,
            "rmse_m": self.rmse,
            "num_cameras": self.num_cameras,
            **self.extra,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "GeoReference":
        """Read a georeference written by :meth:`save`.

        Args:
            path: Path to the ``.json`` file.

        Returns:
            The loaded :class:`GeoReference`.
        """
        d = json.loads(Path(path).read_text())
        return cls(d["crs"], np.asarray(d["origin"]), np.asarray(d["scene_to_geo"]), d["rmse_m"], d["num_cameras"])


def _to_degrees(dms: tuple, ref: str) -> float:
    """Convert EXIF degrees/minutes/seconds to signed decimal degrees.

    Args:
        dms: ``(degrees, minutes, seconds)`` rationals.
        ref: Hemisphere reference (``N``, ``S``, ``E`` or ``W``).

    Returns:
        Decimal degrees (negative for S / W).
    """
    deg = float(dms[0]) + float(dms[1]) / 60.0 + float(dms[2]) / 3600.0
    return -deg if ref in ("S", "W") else deg


def read_gps(path: str | Path) -> tuple[float, float, float] | None:
    """Read the GPS position stored in an image's EXIF.

    Args:
        path: Image path.

    Returns:
        ``(latitude, longitude, altitude)`` in degrees / meters, or ``None`` if
        the image has no GPS tags.
    """
    try:
        with Image.open(path) as im:
            gps = im.getexif().get_ifd(0x8825)
    except Exception:  # noqa: BLE001 - unreadable EXIF means "no GPS"
        return None
    tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}
    if "GPSLatitude" not in tags or "GPSLongitude" not in tags:
        return None
    lat = _to_degrees(tags["GPSLatitude"], tags.get("GPSLatitudeRef", "N"))
    lon = _to_degrees(tags["GPSLongitude"], tags.get("GPSLongitudeRef", "E"))
    alt = float(tags.get("GPSAltitude", 0.0))
    if tags.get("GPSAltitudeRef") in (1, b"\x01"):
        alt = -alt
    return lat, lon, alt


def read_orientation(path: str | Path) -> tuple[float, float | None] | None:
    """Read the camera viewing direction from photo metadata, if present.

    Sources, in order: DJI XMP (``GimbalYawDegree`` / ``GimbalPitchDegree``),
    generic XMP ``Camera:Yaw`` / ``Camera:Pitch``, and EXIF ``GPSImgDirection``
    (heading only).

    Args:
        path: Image path.

    Returns:
        ``(yaw_deg, pitch_deg)`` with yaw clockwise from north and pitch
        negative downwards (``-90`` = nadir); pitch is ``None`` if unknown.
        ``None`` if the photo has no orientation metadata.
    """
    import re

    try:
        with open(path, "rb") as f:
            head = f.read(256_000)
    except OSError:
        return None

    def xmp(*keys: str) -> float | None:
        """Find a numeric XMP attribute or element.

        Args:
            *keys: Attribute names to try.

        Returns:
            The value, or ``None``.
        """
        for key in keys:
            m = re.search(rb"(?:" + key.encode() + rb')="([-+0-9.]+)"', head) or re.search(
                rb"<" + key.encode() + rb">([-+0-9.]+)<", head)
            if m:
                return float(m.group(1))
        return None

    yaw = xmp("drone-dji:GimbalYawDegree", "GimbalYawDegree", "Camera:Yaw")
    pitch = xmp("drone-dji:GimbalPitchDegree", "GimbalPitchDegree", "Camera:Pitch")
    if yaw is None:
        try:
            with Image.open(path) as im:
                gps = im.getexif().get_ifd(0x8825)
            if 17 in gps:  # GPSImgDirection
                yaw = float(gps[17])
        except Exception:  # noqa: BLE001 - no usable EXIF
            return None
    if yaw is None:
        return None
    return yaw % 360.0, pitch


def utm_crs_for(lon: float, lat: float) -> str:
    """WGS84 UTM zone containing a location.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.

    Returns:
        EPSG code string such as ``"EPSG:32630"``.
    """
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Args:
        src: Source points ``[N, 3]``.
        dst: Destination points ``[N, 3]``.

    Returns:
        ``[4, 4]`` matrix ``T`` with ``dst ~= s R src + t``.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = mu_d - s * R @ mu_s
    return T


def georeference(scene: Scene, image_dir: str | Path, crs: str | None = None) -> GeoReference | None:
    """Align a scene to GPS positions read from the original images.

    Images are matched to cameras by file stem, so ``image_dir`` should be the
    folder with the original (EXIF-carrying) photos.

    Args:
        scene: Reconstructed scene (any coordinate frame).
        image_dir: Folder with the original images.
        crs: Target CRS; ``None`` uses the WGS84 UTM zone of the images.

    Returns:
        The :class:`GeoReference`, or ``None`` if fewer than 3 cameras have GPS
        or their positions are degenerate (e.g. all on one line).
    """
    from pyproj import Transformer

    gps_by_stem = {}
    for p in list_images(image_dir):
        g = read_gps(p)
        if g is not None:
            gps_by_stem[p.stem] = g
    pairs = [(c.center, gps_by_stem[Path(c.name).stem]) for c in scene.cameras if Path(c.name).stem in gps_by_stem]
    if len(pairs) < 3:
        logger.info("Georeferencing skipped: %d cameras with GPS", len(pairs))
        return None

    lat, lon, alt = (np.array(v) for v in zip(*[g for _, g in pairs]))
    crs = crs or utm_crs_for(float(np.median(lon)), float(np.median(lat)))
    east, north = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    geo = np.stack([east, north, alt], axis=1)
    origin = np.round(geo.mean(0))
    dst = geo - origin
    src = np.stack([c for c, _ in pairs])

    sv = np.linalg.svd(dst - dst.mean(0), compute_uv=False)
    if sv[1] < 0.05 * sv[0]:
        logger.warning("Georeferencing skipped: GPS positions are (nearly) collinear")
        return None
    T = umeyama(src, dst)
    residual = src @ T[:3, :3].T + T[:3, 3] - dst
    rmse = float(np.sqrt((residual**2).sum(1).mean()))
    logger.info("Georeferenced %d cameras to %s (camera RMSE %.2f m)", len(pairs), crs, rmse)
    return GeoReference(crs, origin, T, rmse, len(pairs))


def local_georeference(scene: Scene) -> GeoReference:
    """Non-georeferenced "map" frame for scenes without GPS.

    The normalized scene (up = -Y, forward = +Z) is mapped to a local Z-up frame
    (x = x, y = z, z = -y) with unit scale and no CRS, so top-down products
    (orthophoto, DSM) can still be made. Units are arbitrary (SfM units).

    Args:
        scene: Normalized scene (see :func:`geosplat.scene.normalize_scene`).

    Returns:
        A :class:`GeoReference` with ``crs = None`` and ``origin = 0``.
    """
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    return GeoReference(None, np.zeros(3), T, 0.0, len(scene.cameras), {"note": "local frame, no CRS"})  # type: ignore[arg-type]


def gravity_in_scene(geo: GeoReference) -> tuple[np.ndarray, np.ndarray]:
    """Down and north directions of the geo frame expressed in scene coordinates.

    Args:
        geo: Georeference of the scene.

    Returns:
        Tuple ``(down, north)`` of unit vectors ``[3]``.
    """
    R = geo.rotation
    return -R.T @ np.array([0.0, 0.0, 1.0]), R.T @ np.array([0.0, 1.0, 0.0])


def read_gcps(path: str | Path) -> dict[str, np.ndarray]:
    """Read ground control points from a ``name,x,y,z`` text file.

    The delimiter (``,`` ``;`` tab or spaces) is detected automatically, and
    header or other non-numeric lines are skipped.

    Args:
        path: CSV / text file path.

    Returns:
        Mapping from point name to ``[3]`` CRS coordinates.
    """
    text = Path(path).read_text(errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t ")
        rows = csv.reader(text.splitlines(), dialect)
    except csv.Error:
        rows = (line.replace(";", ",").replace("\t", ",").split(",") for line in text.splitlines())
    gcps = {}
    for row in rows:
        row = [c.strip() for c in row if c.strip()]
        if len(row) < 4:
            continue
        try:
            gcps[row[0]] = np.array([float(v) for v in row[1:4]])
        except ValueError:
            continue  # header or comment line
    return gcps


def find_gcp_file(image_dir: str | Path, min_points: int = 3) -> Path | None:
    """Find a ground-control-point CSV (``name,x,y,z``) next to the photos.

    Any ``*.csv`` in ``image_dir`` with at least ``min_points`` numeric rows
    qualifies. Coordinates are assumed to be in the output CRS (the photos'
    UTM zone unless another CRS is configured).

    Args:
        image_dir: Folder with the photos.
        min_points: Minimum number of valid points.

    Returns:
        Path of the first matching CSV, or ``None``.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        return None
    for csv_path in sorted(image_dir.glob("*.csv")):
        try:
            if len(read_gcps(csv_path)) >= min_points:
                logger.info("Ground control points found: %s", csv_path.name)
                return csv_path
        except (OSError, UnicodeDecodeError):
            continue
    return None
