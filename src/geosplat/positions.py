"""Photo positions from every common source: EXIF, XMP sidecars, external files, GPX tracks.

:func:`photo_positions` returns one WGS84 position per photo, combining:

1. **Positions file** (overrides EXIF; usually RTK/PPK): OpenDroneMap ``geo.txt``,
   CSV / TXT with photo names and coordinates (Metashape, Pix4D, PPK software...),
   point layers (``.gpkg``, ``.geojson``, ``.shp``, ``.kml``) with a photo-name field,
   or DJI RTK ``*.MRK`` files.
2. **XMP sidecars** (``IMG_1.xmp`` or ``IMG_1.JPG.xmp``; override EXIF).
3. **EXIF GPS** inside the photos.
4. **GPX tracks**: photos still without a position are located by their capture time.

Files are searched in the photo folder and its parent folder.
"""

from __future__ import annotations

import csv
import functools
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image

from .geo import read_gps, read_orientation
from .sfm import list_images

logger = logging.getLogger("geosplat")

TABLE_EXTENSIONS = (".csv", ".txt")
VECTOR_EXTENSIONS = (".gpkg", ".geojson", ".json", ".shp", ".kml", ".fgb", ".gml")
NAME_COLS = ("name", "image", "image_name", "imagename", "filename", "file", "photo", "label", "img", "#label")
LAT_COLS = ("lat", "latitude", "lat_deg", "gps_lat", "latitud")
LON_COLS = ("lon", "long", "lng", "longitude", "lon_deg", "gps_lon", "longitud")
X_COLS = ("x", "easting", "east", "e", "x_m", "este")
Y_COLS = ("y", "northing", "north", "n", "y_m", "norte")
Z_COLS = ("z", "alt", "altitude", "height", "h", "elevation", "ellh", "ele", "z_m", "altitud", "cota")
YAW_COLS = ("yaw", "heading", "kappa")
PITCH_COLS = ("pitch",)


@dataclass
class PhotoPosition:
    """Where a photo was taken.

    Attributes:
        lat: Latitude (WGS84 degrees).
        lon: Longitude (WGS84 degrees).
        alt: Altitude in meters (``nan`` if unknown).
        yaw: Heading in degrees clockwise from north, if known.
        pitch: Gimbal pitch in degrees (``-90`` = nadir), if known.
        source: Where the position came from (e.g. ``"exif"``, ``"geo.txt"``).
    """

    lat: float
    lon: float
    alt: float = float("nan")
    yaw: float | None = None
    pitch: float | None = None
    source: str = "exif"


def _key(name: str) -> str:
    """Normalize a photo reference for matching (file name without extension, lower case).

    Args:
        name: File name or stem.

    Returns:
        Matching key.
    """
    return Path(str(name).strip().strip('"')).stem.lower()


def _to_wgs84(x: np.ndarray, y: np.ndarray, crs: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Convert coordinates to longitude / latitude.

    Args:
        x: Eastings or longitudes.
        y: Northings or latitudes.
        crs: Source CRS (``None`` or geographic: returned unchanged).

    Returns:
        Tuple ``(lon, lat)``.
    """
    from pyproj import CRS, Transformer

    if crs is None or CRS.from_user_input(crs).is_geographic:
        return x, y
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
    return np.asarray(lon), np.asarray(lat)


def _parse_crs_line(line: str) -> str | None:
    """Parse the CRS header line of an OpenDroneMap ``geo.txt``.

    Accepts ``EPSG:xxxx``, PROJ strings and ODM's ``WGS84 UTM 32N`` form.

    Args:
        line: First line of the file.

    Returns:
        CRS string understood by pyproj, or ``None`` if the line is not a CRS.
    """
    from pyproj import CRS

    text = line.strip()
    m = re.fullmatch(r"WGS84\s+UTM\s+(\d{1,2})\s*([NS])", text, re.IGNORECASE)
    if m:
        return f"EPSG:{(32600 if m.group(2).upper() == 'N' else 32700) + int(m.group(1))}"
    try:
        CRS.from_user_input(text)
        return text
    except Exception:  # noqa: BLE001 - not a CRS line
        return None


def _rows(path: Path) -> list[list[str]]:
    """Split a text table into rows (delimiter sniffed: comma, semicolon, tab or spaces).

    Args:
        path: Text file.

    Returns:
        Non-empty rows of stripped cells (comment lines starting with ``#`` kept, see callers).
    """
    text = path.read_text(errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and all("\t" in ln for ln in lines[:20]):          # tab tables (headers may contain spaces)
        rows = [ln.split("\t") for ln in lines]
        return [[c.strip() for c in r if c.strip()] for r in rows if any(c.strip() for c in r)]
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines[:20]), delimiters=",;\t ")
        rows = list(csv.reader(lines, dialect))
    except csv.Error:
        rows = [re.split(r"[,;\t ]+", ln.strip()) for ln in lines]
    return [[c.strip() for c in r if c.strip()] for r in rows if any(c.strip() for c in r)]


def norm_header(cell: str) -> str:
    """Normalize a table header cell for matching column names.

    ``"# File"`` -> ``"file"``, ``"WGS84_Lat"`` -> ``"lat"``, ``"Ell.H(m)"`` -> ``"ellh"``,
    ``"Altitude (m)"`` -> ``"altitude"``.

    Args:
        cell: Header text.

    Returns:
        Lower-case name without comment marks, units, dots or CRS prefixes.
    """
    c = re.sub(r"[\(\[].*?[\)\]]", "", cell.lower()).strip().lstrip("#").strip()
    c = c.replace(".", "").replace(" ", "_")
    return re.sub(r"^(wgs_?84|etrs_?89|gps|gnss)_", "", c)


def _num(v: str) -> float | None:
    """Parse a number, returning ``None`` if it is not one.

    Args:
        v: Text.

    Returns:
        Float value or ``None``.
    """
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_table(path: str | Path, names: set[str], crs: str | None = None) -> dict[str, PhotoPosition]:
    """Read photo positions from a CSV / TXT table (ODM ``geo.txt`` or generic).

    Args:
        path: Table file.
        names: Matching keys of the photos (see :func:`_key`); rows for other
            names are ignored.
        crs: CRS of projected coordinates when the file does not state it.

    Returns:
        ``{photo_key: PhotoPosition}`` (empty if the file is not a positions table).
    """
    path = Path(path)
    rows = _rows(path)
    if not rows:
        return {}
    source = path.name
    file_crs = None
    first = " ".join(rows[0])
    if len(rows[0]) <= 4 and _key(rows[0][0]) not in names and _num(rows[0][-1]) is None:
        file_crs = _parse_crs_line(first)          # ODM geo.txt header line
        if file_crs:
            rows = rows[1:]
    if rows and rows[0][0] == "#":                  # "# name lat lon" header split at the "#"
        rows[0] = rows[0][1:]
    header = [norm_header(c) for c in rows[0]] if rows and _num(rows[0][-1]) is None else None
    if header:
        rows = rows[1:]

    def col(options: tuple[str, ...]) -> int | None:
        """Index of the first header column matching ``options``.

        Args:
            options: Accepted column names.

        Returns:
            Column index or ``None``.
        """
        return next((header.index(o) for o in options if header and o in header), None)

    if header:
        i_name = col(NAME_COLS) if col(NAME_COLS) is not None else 0
        i_lat, i_lon, i_x, i_y = col(LAT_COLS), col(LON_COLS), col(X_COLS), col(Y_COLS)
        geographic = i_lat is not None and i_lon is not None
        i_a, i_b = (i_lon, i_lat) if geographic else (i_x, i_y)
        i_z, i_yaw, i_pitch = col(Z_COLS), col(YAW_COLS), col(PITCH_COLS)
        if i_a is None or i_b is None:
            return {}
    else:                                           # name x y [z] [yaw pitch roll] (ODM / Metashape order)
        i_name, i_a, i_b, i_z, i_yaw, i_pitch = 0, 1, 2, 3, 4, 5
        geographic = None

    keys, a, b, z, yaw, pitch = [], [], [], [], [], []
    for r in rows:
        if len(r) <= max(i_name, i_a, i_b) or _key(r[i_name]) not in names:
            continue
        va, vb = _num(r[i_a]), _num(r[i_b])
        if va is None or vb is None:
            continue
        keys.append(_key(r[i_name]))
        a.append(va)
        b.append(vb)
        z.append(_num(r[i_z]) if i_z is not None and i_z < len(r) and _num(r[i_z]) is not None else np.nan)
        yaw.append(_num(r[i_yaw]) if i_yaw is not None and i_yaw < len(r) else None)
        pitch.append(_num(r[i_pitch]) if i_pitch is not None and i_pitch < len(r) else None)
    if not keys:
        return {}
    a_arr, b_arr = np.array(a), np.array(b)
    src_crs = file_crs or crs
    if src_crs is None:
        looks_lonlat = np.all(np.abs(a_arr) <= 180) and np.all(np.abs(b_arr) <= 90)
        if geographic is False or not looks_lonlat:
            logger.warning("%s has projected coordinates but no CRS; set positions_crs (e.g. EPSG:25830).", path.name)
            return {}
    lon, lat = _to_wgs84(a_arr, b_arr, src_crs)
    if file_crs is None and not header:              # ODM-style rows without header: no reliable angles
        yaw = [None] * len(keys)
        pitch = [None] * len(keys)
    elif header and "roll" in header:
        # Yaw/pitch/roll of the aircraft (Metashape, Pix4D): pitch ~0 means a level, down-looking camera.
        # Convert to the gimbal convention used here (-90 = nadir).
        known = [p for p in pitch if p is not None]
        if known and float(np.median(np.abs(known))) < 30:
            pitch = [None if p is None else p - 90.0 for p in pitch]
    return {k: PhotoPosition(float(lat[i]), float(lon[i]), float(z[i]), yaw[i], pitch[i], source)
            for i, k in enumerate(keys)}


def read_vector(path: str | Path, names: set[str]) -> dict[str, PhotoPosition]:
    """Read photo positions from a point layer with a photo-name field.

    Args:
        path: Vector file (``.gpkg``, ``.geojson``, ``.shp``, ``.kml``...).
        names: Matching keys of the photos.

    Returns:
        ``{photo_key: PhotoPosition}`` (empty if no feature names a photo).
    """
    import pyogrio
    import shapely

    meta, _, geoms, field_data = pyogrio.raw.read(str(path))
    fields = {str(n).lower(): np.asarray(v) for n, v in zip(meta["fields"], field_data)}
    name_col = next((fields[f] for f in NAME_COLS if f in fields), None)
    if name_col is None:
        name_col = next((v for v in fields.values() if any(_key(x) in names for x in v[:50])), None)
    if name_col is None:
        return {}
    z_col = next((fields[f] for f in Z_COLS if f in fields), None)
    yaw_col = next((fields[f] for f in YAW_COLS if f in fields), None)
    keys, xs, ys, zs, yaws = [], [], [], [], []
    for i, g in enumerate(shapely.from_wkb(geoms)):
        k = _key(name_col[i])
        if g is None or g.is_empty or k not in names:
            continue
        g = g if g.geom_type == "Point" else g.centroid
        keys.append(k)
        xs.append(g.x)
        ys.append(g.y)
        zs.append(g.z if g.has_z else (float(z_col[i]) if z_col is not None else np.nan))
        yaws.append(float(yaw_col[i]) if yaw_col is not None else None)
    if not keys:
        return {}
    lon, lat = _to_wgs84(np.array(xs), np.array(ys), meta.get("crs") or "EPSG:4326")
    return {k: PhotoPosition(float(lat[i]), float(lon[i]), float(zs[i]), yaws[i], None, Path(path).name)
            for i, k in enumerate(keys)}


def read_mrk(path: str | Path, photos: list[Path]) -> dict[str, PhotoPosition]:
    """Read DJI RTK ``*.MRK`` timestamp files (photo index, latitude, longitude, ellipsoidal height).

    Photos are matched by the number at the end of their file name
    (``DJI_20240411_0012.JPG`` is index 12).

    Args:
        path: MRK file.
        photos: Photo paths.

    Returns:
        ``{photo_key: PhotoPosition}``.
    """
    by_index = {}
    for p in photos:
        m = re.search(r"(\d+)$", p.stem)
        if m:
            by_index.setdefault(int(m.group(1)), p)
    out = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        idx = re.match(r"\s*(\d+)\s", line)
        lat = re.search(r"([-+]?\d+\.\d+)\s*,\s*Lat", line)
        lon = re.search(r"([-+]?\d+\.\d+)\s*,\s*Lon", line)
        h = re.search(r"([-+]?\d+(?:\.\d+)?)\s*,\s*Ellh", line)
        if idx and lat and lon and int(idx.group(1)) in by_index:
            p = by_index[int(idx.group(1))]
            out[_key(p.name)] = PhotoPosition(float(lat.group(1)), float(lon.group(1)),
                                              float(h.group(1)) if h else float("nan"), source=Path(path).name)
    return out


def read_xmp_sidecar(photo: Path) -> PhotoPosition | None:
    """Read GPS (and DJI gimbal angles) from an XMP sidecar next to a photo.

    Args:
        photo: Photo path; ``<stem>.xmp`` or ``<name>.xmp`` is looked for.

    Returns:
        Position, or ``None`` if there is no sidecar with GPS.
    """
    for side in (photo.with_suffix(".xmp"), photo.with_suffix(".XMP"), Path(str(photo) + ".xmp")):
        if not side.is_file():
            continue
        text = side.read_text(errors="replace")

        def coord(tag: str) -> float | None:
            """Parse an XMP GPS coordinate (``40,36.8078N`` or decimal).

            Args:
                tag: Tag name, e.g. ``GPSLatitude``.

            Returns:
                Signed decimal degrees, or ``None``.
            """
            m = re.search(tag + r'\s*=\s*"([^"]+)"', text) or re.search(r"<[^>]*" + tag + r">([^<]+)<", text)
            if not m:
                return None
            v = m.group(1).strip()
            dm = re.fullmatch(r"(\d+),(\d+(?:\.\d+)?)([NSEW])", v)
            if dm:
                deg = float(dm.group(1)) + float(dm.group(2)) / 60
                return -deg if dm.group(3) in "SW" else deg
            return _num(v)

        lat, lon = coord("GPSLatitude"), coord("GPSLongitude")
        if lat is None or lon is None:
            continue
        alt_m = re.search(r'GPSAltitude\s*=\s*"([^"]+)"', text)
        alt = float("nan")
        if alt_m:
            v = alt_m.group(1)
            alt = float(v.split("/")[0]) / float(v.split("/")[1]) if "/" in v else (_num(v) or float("nan"))
        yaw = re.search(r'GimbalYawDegree\s*=\s*"([-+0-9.]+)"', text)
        pitch = re.search(r'GimbalPitchDegree\s*=\s*"([-+0-9.]+)"', text)
        return PhotoPosition(lat, lon, alt, float(yaw.group(1)) % 360 if yaw else None,
                             float(pitch.group(1)) if pitch else None, side.name)
    return None


def _capture_time(photo: Path) -> datetime | None:
    """Capture time of a photo from EXIF (UTC if the EXIF offset tag is present, else naive).

    Args:
        photo: Photo path.

    Returns:
        Datetime or ``None``.
    """
    try:
        with Image.open(photo) as im:
            exif = im.getexif()
            sub = exif.get_ifd(0x8769)
    except Exception:  # noqa: BLE001
        return None
    tags = {ExifTags.TAGS.get(k, k): v for k, v in {**exif, **sub}.items()}
    raw = tags.get("DateTimeOriginal") or tags.get("DateTime")
    if not raw:
        return None
    try:
        t = datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    off = tags.get("OffsetTimeOriginal") or tags.get("OffsetTime")
    if off and re.fullmatch(r"[+-]\d{2}:\d{2}", str(off)):
        sign = 1 if str(off)[0] == "+" else -1
        delta = timedelta(hours=int(str(off)[1:3]), minutes=int(str(off)[4:6]))
        t = (t - sign * delta).replace(tzinfo=timezone.utc)
    return t


def read_gpx(path: str | Path, photos: list[Path], time_offset_s: float | None = None) -> dict[str, PhotoPosition]:
    """Locate photos on a GPX track by their capture time (linear interpolation).

    Without a UTC offset in the EXIF, the camera clock offset is taken from
    ``time_offset_s`` or estimated as the whole number of hours (-12..+14) that
    puts the most photos inside the track's time span.

    Args:
        path: GPX file.
        photos: Photos to locate.
        time_offset_s: Seconds to add to camera times to get UTC (``None``: auto).

    Returns:
        ``{photo_key: PhotoPosition}`` for photos within the track (±60 s).
    """
    text = Path(path).read_text(errors="replace")
    pts = []
    for m in re.finditer(r'<trkpt[^>]*lat="([-0-9.]+)"[^>]*lon="([-0-9.]+)"[^>]*>(.*?)</trkpt>', text, re.S):
        tm = re.search(r"<time>([^<]+)</time>", m.group(3))
        if not tm:
            continue
        ele = re.search(r"<ele>([^<]+)</ele>", m.group(3))
        t = datetime.fromisoformat(tm.group(1).strip().replace("Z", "+00:00"))
        t = t if t.tzinfo else t.replace(tzinfo=timezone.utc)
        pts.append((t.timestamp(), float(m.group(1)), float(m.group(2)), float(ele.group(1)) if ele else np.nan))
    if len(pts) < 2:
        return {}
    pts.sort()
    tt, la, lo, el = (np.array(v) for v in zip(*pts))
    times = {p: _capture_time(p) for p in photos}
    naive = [t.replace(tzinfo=timezone.utc).timestamp() for t in times.values() if t is not None and t.tzinfo is None]
    offset = time_offset_s
    if offset is None and naive:
        naive_arr = np.array(naive)
        offset = max((h * 3600 for h in range(-12, 15)),
                     key=lambda o: (np.sum((naive_arr - o >= tt[0] - 60) & (naive_arr - o <= tt[-1] + 60)), -abs(o)))
        if offset:
            logger.info("GPX: camera clock offset estimated as %+d h", offset // 3600)
    out = {}
    for p, t in times.items():
        if t is None:
            continue
        ts = t.timestamp() if t.tzinfo else t.replace(tzinfo=timezone.utc).timestamp() - (offset or 0)
        if tt[0] - 60 <= ts <= tt[-1] + 60:
            out[_key(p.name)] = PhotoPosition(float(np.interp(ts, tt, la)), float(np.interp(ts, tt, lo)),
                                              float(np.interp(ts, tt, el)), source=Path(path).name)
    return out


def _candidate_files(image_dir: Path) -> list[Path]:
    """Files in the photo folder and its parent that may hold positions.

    Args:
        image_dir: Photo folder.

    Returns:
        Candidate files (photos excluded).
    """
    exts = TABLE_EXTENSIONS + VECTOR_EXTENSIONS + (".mrk", ".gpx")
    out = []
    for d in dict.fromkeys([image_dir, image_dir.parent]):
        if d.is_dir():
            out += sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)
    return out


def read_positions_file(path: str | Path, photos: list[Path], crs: str | None = None,
                        gpx_time_offset_s: float | None = None) -> dict[str, PhotoPosition]:
    """Read photo positions from any supported external file.

    Args:
        path: ``geo.txt`` / CSV / TXT, vector file, DJI ``.MRK`` or ``.gpx``.
        photos: Photo paths.
        crs: CRS of projected coordinates in tables without a CRS.
        gpx_time_offset_s: Camera-to-UTC offset for GPX matching (``None``: auto).

    Returns:
        ``{photo_key: PhotoPosition}`` (empty if the file holds no positions for these photos).
    """
    path = Path(path)
    names = {_key(p.name) for p in photos}
    suffix = path.suffix.lower()
    try:
        if suffix == ".mrk":
            return read_mrk(path, photos)
        if suffix == ".gpx":
            return read_gpx(path, photos, gpx_time_offset_s)
        if suffix in VECTOR_EXTENSIONS:
            return read_vector(path, names)
        if suffix in TABLE_EXTENSIONS:
            return read_table(path, names, crs)
    except Exception as exc:  # noqa: BLE001 - unreadable or unrelated file
        logger.debug("Ignoring %s: %s", path.name, exc)
    return {}


def is_positions_file(path: str | Path, image_dir: str | Path) -> bool:
    """Whether a file lists positions of the photos (so it must not be taken for GCPs).

    Args:
        path: Candidate file.
        image_dir: Photo folder.

    Returns:
        ``True`` if at least half of the photos (or 3) are found in it.
    """
    from .sources import scan_dataset

    try:
        photos = scan_dataset(image_dir).photos
    except FileNotFoundError:
        return False
    if not photos:                                 # no photos: nothing to list positions for
        return False
    n = len(read_positions_file(path, photos))
    return n >= max(3, len(photos) // 2) or (n >= 1 and n == len(photos))


@functools.lru_cache(maxsize=64)
def _cached_positions(image_dir: str, stamp: tuple, positions_file: str | None, crs: str | None,
                      gpx_offset: float | None, extra: tuple[str, ...] = ()) -> dict[str, PhotoPosition]:
    """Cached implementation of :func:`photo_positions` (see there).

    Args:
        image_dir: Photo folder.
        stamp: Folder modification fingerprint (invalidates the cache).
        positions_file: Explicit positions file or ``None``.
        crs: CRS for projected tables without one.
        gpx_offset: Camera-to-UTC offset for GPX.
        extra: Additional candidate files (e.g. side files elsewhere in the dataset).

    Returns:
        ``{photo_key: PhotoPosition}``.
    """
    folder = Path(image_dir)
    photos = list_images(folder)
    out: dict[str, PhotoPosition] = {}
    for p in photos:                                                     # 3. EXIF
        g = read_gps(p)
        if g is not None:
            o = read_orientation(p)
            out[_key(p.name)] = PhotoPosition(g[0], g[1], g[2], o[0] if o else None, o[1] if o else None, "exif")
    for p in photos:                                                     # 2. XMP sidecars
        side = read_xmp_sidecar(p)
        if side is not None:
            out[_key(p.name)] = side
    files = [Path(positions_file)] if positions_file else list(dict.fromkeys(
        _candidate_files(folder) + [Path(f) for f in extra]))
    gpx = [f for f in files if f.suffix.lower() == ".gpx"]
    for f in [f for f in files if f.suffix.lower() != ".gpx"]:          # 1. positions files (override)
        found = read_positions_file(f, photos, crs)
        if found and (positions_file or len(found) >= max(3, len(photos) // 2) or len(found) == len(photos)):
            for k, v in found.items():
                old = out.get(k)
                if old is not None and v.yaw is None:                    # keep EXIF orientation
                    v.yaw, v.pitch = old.yaw, old.pitch
                out[k] = v
            logger.info("Photo positions from %s (%d photos)", f.name, len(found))
    missing = [p for p in photos if _key(p.name) not in out]
    for f in gpx:                                                        # 4. GPX for the rest
        if not missing:
            break
        found = read_gpx(f, missing, gpx_offset)
        out.update(found)
        if found:
            logger.info("Photo positions from GPX %s (%d photos)", f.name, len(found))
        missing = [p for p in missing if _key(p.name) not in out]
    return out


def photo_positions(
    image_dir: str | Path,
    positions_file: str | Path | None = None,
    crs: str | None = None,
    gpx_time_offset_s: float | None = None,
    extra_files: list[Path] | tuple[Path, ...] = (),
) -> dict[str, PhotoPosition]:
    """Position of every photo that has one, from all available sources.

    Args:
        image_dir: Photo folder.
        positions_file: Explicit positions file (otherwise files in the photo folder
            and its parent are detected automatically).
        crs: CRS of projected coordinates in tables that do not state it.
        gpx_time_offset_s: Seconds to add to camera times to get UTC for GPX matching
            (``None``: from EXIF offset tags or estimated).
        extra_files: More candidate positions files (e.g. a ``logs`` folder elsewhere
            in the dataset); files that do not list these photos are ignored.

    Returns:
        ``{photo_key: PhotoPosition}``; keys are file names without extension, lower
        case (use :func:`position_of` to look up a photo path).
    """
    folder = Path(image_dir).resolve()
    if not folder.is_dir():
        return {}
    stamp = tuple(sorted((p.name, p.stat().st_mtime) for d in {folder, folder.parent} for p in d.iterdir()
                         if p.is_file() and p.suffix.lower() in TABLE_EXTENSIONS + VECTOR_EXTENSIONS
                         + (".mrk", ".gpx", ".xmp")))
    extra = tuple(sorted(str(Path(f).resolve()) for f in extra_files
                         if Path(f).suffix.lower() in TABLE_EXTENSIONS + VECTOR_EXTENSIONS + (".mrk", ".gpx")))
    stamp += tuple((f, Path(f).stat().st_mtime) for f in extra if Path(f).exists())
    return _cached_positions(str(folder), stamp, str(positions_file) if positions_file else None, crs,
                             gpx_time_offset_s, extra)


def position_of(photo: str | Path, positions: dict[str, PhotoPosition]) -> PhotoPosition | None:
    """Look up the position of a photo path.

    Args:
        photo: Photo path or name.
        positions: Result of :func:`photo_positions`.

    Returns:
        The position or ``None``.
    """
    return positions.get(_key(Path(photo).name))
