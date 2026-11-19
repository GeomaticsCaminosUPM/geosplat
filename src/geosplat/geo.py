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
            ``None`` uses a GCP file found in the photo folder, if any (vector
            files with a CRS such as ``.gpkg`` / ``.geojson`` / ``.shp``, or CSV).
        gcps_crs: CRS of the GCP coordinates (e.g. ``"EPSG:25830"``), needed only
            for CSV files; ``None`` reads it from vector files or infers it from
            the photo positions.
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
    gcps_crs: str | None = None


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


GCP_VECTOR_EXTENSIONS = (".gpkg", ".geojson", ".json", ".shp", ".gml", ".kml", ".fgb")
GCP_TEXT_EXTENSIONS = (".csv", ".txt")
_NAME_FIELDS = ("name", "id", "label", "point", "code", "nombre", "punto", "diana", "target")
_Z_FIELDS = ("z", "h", "height", "elev", "elevation", "alt", "altitude", "cota", "altura")


def read_gcp_file(path: str | Path) -> tuple[dict[str, np.ndarray], str | None]:
    """Read ground control points from a CSV or a vector file (GeoPackage, Shapefile, GeoJSON...).

    Vector files: one point per feature; the name comes from a field such as
    ``name`` / ``id`` / ``label`` (or the feature index) and the height from a 3D
    geometry or a field such as ``z`` / ``h`` / ``height`` (``nan`` if absent).

    Args:
        path: GCP file.

    Returns:
        Tuple ``(points, crs)``: ``{name: [x, y, z]}`` in the file's coordinates
        and the file's CRS (``None`` for CSV / text files, which carry no CRS).
    """
    path = Path(path)
    if path.suffix.lower() not in GCP_VECTOR_EXTENSIONS:
        return read_gcps(path), None
    import pyogrio
    import shapely

    meta, _, geoms, field_data = pyogrio.raw.read(str(path))
    fields = {str(n).lower(): np.asarray(v) for n, v in zip(meta["fields"], field_data)}
    name_col = next((fields[f] for f in _NAME_FIELDS if f in fields), None)
    z_col = next((fields[f] for f in _Z_FIELDS if f in fields), None)
    points = {}
    for i, g in enumerate(shapely.from_wkb(geoms)):
        if g is None or g.is_empty:
            continue
        g = g if g.geom_type == "Point" else g.centroid
        z = g.z if g.has_z else (float(z_col[i]) if z_col is not None else np.nan)
        name = str(name_col[i]) if name_col is not None else str(i + 1)
        points[name] = np.array([g.x, g.y, z], dtype=np.float64)
    return points, meta.get("crs")


def _photo_lonlat(image_dir: str | Path) -> np.ndarray:
    """GPS positions of the photos in a folder.

    Args:
        image_dir: Folder with photos.

    Returns:
        Array ``[N, 2]`` of ``(lon, lat)`` (empty if no photo has GPS).
    """
    pos = [g for g in (read_gps(p) for p in list_images(image_dir)) if g is not None]
    return np.array([[lon, lat] for lat, lon, _ in pos]).reshape(-1, 2)


def infer_gcp_crs(points: dict[str, np.ndarray], photo_lonlat: np.ndarray) -> tuple[str | None, float]:
    """Guess the CRS of GCP coordinates from where the photos were taken.

    Candidates are geographic lon/lat and the UTM zones around the photos (WGS84;
    also ETRS89 in Europe and NAD83 in North America). The winner places the most
    GCPs inside the photographed area (convex hull of the photo positions,
    enlarged by 30 % of its size, at least 50 m).

    Args:
        points: ``{name: [x, y, z]}`` as read from the file.
        photo_lonlat: Photo positions ``[N, 2]`` (lon, lat).

    Returns:
        Tuple ``(crs, fraction_inside)``; ``crs`` is ``None`` if no candidate puts at
        least half of the points inside the area.
    """
    import shapely
    from pyproj import Transformer

    if len(points) == 0 or len(photo_lonlat) == 0:
        return None, 0.0
    xy = np.array([p[:2] for p in points.values()])
    lon, lat = float(np.median(photo_lonlat[:, 0])), float(np.median(photo_lonlat[:, 1]))
    zone = int((lon + 180) // 6) + 1
    north = lat >= 0
    candidates = ["EPSG:4326"] if np.all(np.abs(xy[:, 0]) <= 180) and np.all(np.abs(xy[:, 1]) <= 90) else []
    for z in (zone, zone - 1, zone + 1):
        if not 1 <= z <= 60:
            continue
        if north and -32 <= lon <= 45 and lat >= 27 and 28 <= z <= 38:
            candidates.append(f"EPSG:{25800 + z}")          # ETRS89 / UTM (Europe)
        candidates.append(f"EPSG:{(32600 if north else 32700) + z}")   # WGS84 / UTM
        if north and -170 <= lon <= -50 and 1 <= z <= 23:
            candidates.append(f"EPSG:{26900 + z}")          # NAD83 / UTM (North America)
    best, best_score = None, 0.0
    for crs in candidates:
        if crs == "EPSG:4326":
            px, py = photo_lonlat[:, 0], photo_lonlat[:, 1]
            hull = shapely.MultiPoint(np.c_[px, py]).convex_hull
            size = max(np.ptp(px), np.ptp(py))
            area = hull.buffer(max(0.3 * size, 50 / 111_320))
        else:
            t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            px, py = t.transform(photo_lonlat[:, 0], photo_lonlat[:, 1])
            hull = shapely.MultiPoint(np.c_[px, py]).convex_hull
            area = hull.buffer(max(0.3 * max(np.ptp(px), np.ptp(py)), 50.0))
        score = float(np.mean(shapely.contains_xy(area, xy[:, 0], xy[:, 1])))
        if score > best_score:  # ties keep the earlier (preferred) candidate
            best, best_score = crs, score
    return (best, best_score) if best_score >= 0.5 else (None, best_score)


def find_gcp_file(image_dir: str | Path, min_points: int = 3) -> Path | None:
    """Find a ground-control-point file next to the photos.

    Vector files with a CRS (``.gpkg``, ``.geojson``, ``.shp``...) are preferred
    over CSV / text files (``name,x,y,z``, header optional).

    Args:
        image_dir: Folder with the photos.
        min_points: Minimum number of valid points.

    Returns:
        Path of the first matching file, or ``None``.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        return None
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in GCP_VECTOR_EXTENSIONS)
    files += sorted(p for p in image_dir.iterdir() if p.suffix.lower() in GCP_TEXT_EXTENSIONS)
    for path in files:
        try:
            if len(read_gcp_file(path)[0]) >= min_points:
                logger.info("Ground control points found: %s", path.name)
                return path
        except Exception:  # noqa: BLE001 - unreadable / unrelated file
            continue
    return None


def gcp_setup(image_dir: str | Path, config: GeoConfig) -> tuple[Path | None, str | None]:
    """Find the GCP file and its CRS, before georeferencing.

    The CRS is, in order: ``config.gcps_crs``, the file's own CRS (vector files),
    or a guess from the photo positions (:func:`infer_gcp_crs`). If no output CRS
    is configured and the GCP CRS is projected, it becomes the output CRS, so all
    products are delivered in the survey's coordinate system.

    Args:
        image_dir: Folder with the photos.
        config: Georeferencing options; ``config.crs`` may be filled in.

    Returns:
        Tuple ``(gcp_file, gcp_crs)`` (``None`` values when there is no GCP file
        or the CRS could not be determined).
    """
    from pyproj import CRS

    path = Path(config.gcps) if config.gcps else find_gcp_file(image_dir)
    if path is None:
        return None, None
    points, file_crs = read_gcp_file(path)
    crs = config.gcps_crs or file_crs
    if crs is None:
        crs, score = infer_gcp_crs(points, _photo_lonlat(image_dir))
        if crs:
            logger.info("GCP CRS inferred as %s (%.0f %% of points inside the photographed area)", crs, 100 * score)
        else:
            logger.warning("Could not infer the CRS of %s; set it explicitly (gcps_crs / GCPS_CRS).", path.name)
    if crs and config.crs is None and CRS.from_user_input(crs).is_projected:
        config.crs = CRS.from_user_input(crs).to_string()
    return path, crs


def load_gcps(path: str | Path, target_crs: str | None, gcp_crs: str | None = None) -> dict[str, np.ndarray]:
    """Read a GCP file and convert its coordinates to the output CRS.

    Args:
        path: GCP file (CSV or vector).
        target_crs: CRS of the georeference (``geo.crs``).
        gcp_crs: CRS of the file's coordinates (from :func:`gcp_setup`); ``None``
            assumes they are already in ``target_crs``.

    Returns:
        ``{name: [x, y, z]}`` in ``target_crs`` (heights unchanged; ``nan`` if unknown).
    """
    from pyproj import CRS, Transformer

    points, file_crs = read_gcp_file(path)
    src = gcp_crs or file_crs
    if not points or src is None or target_crs is None or CRS.from_user_input(src) == CRS.from_user_input(target_crs):
        return points
    t = Transformer.from_crs(src, target_crs, always_xy=True)
    names = list(points)
    xyz = np.array([points[n] for n in names])
    x, y = t.transform(xyz[:, 0], xyz[:, 1])
    return {n: np.array([x[i], y[i], xyz[i, 2]]) for i, n in enumerate(names)}
