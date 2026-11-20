"""Interactive maps for notebooks: photo locations with polygon selection, orthophoto overlay.

Requires ``ipyleaflet`` (``pip install ipyleaflet``; on Colab also call
``google.colab.output.enable_custom_widget_manager()``).

Basemaps: Google Satellite and Google Hybrid are both added as switchable base
layers (layer control, top right); ``"esri"`` (Esri World Imagery) is also available.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np

from .geo import read_orientation
from .positions import photo_positions, position_of
from .sfm import list_images

MAX_ZOOM = 24          # map zoom limit; basemap tiles are upscaled beyond their native zoom
NATIVE_ZOOM = 20       # deepest zoom level fetched from the tile servers

BASEMAPS = {
    "google_hybrid": ("https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", "Google"),
    "google_satellite": ("https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", "Google"),
    "esri": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Esri, Maxar, Earthstar Geographics",
    ),
}


def _basemap(name: str) -> object:
    """Create an ipyleaflet tile layer for a named basemap.

    Args:
        name: Key of :data:`BASEMAPS`.

    Returns:
        An ``ipyleaflet.TileLayer`` usable as a base layer.
    """
    from ipyleaflet import TileLayer

    url, attribution = BASEMAPS[name]
    label = name.replace("_", " ").title()
    return TileLayer(url=url, attribution=attribution, name=label, max_zoom=MAX_ZOOM, max_native_zoom=NATIVE_ZOOM, base=True)


def _add_other_google_layers(m: object, current: str) -> None:
    """Add the Google base layer not used as default, plus a layer switcher.

    Args:
        m: ``ipyleaflet.Map``.
        current: Name of the basemap already shown.
    """
    from ipyleaflet import LayersControl

    for name in ("google_satellite", "google_hybrid"):
        if name != current:
            m.add(_basemap(name))
    m.add(LayersControl(position="topright"))


def label_icon(text: str, color: str = "#ff3333", symbol: str = "") -> object:
    """A map icon made of an optional symbol and a text label.

    Args:
        text: Label text (e.g. a GCP name).
        color: Text / symbol color.
        symbol: Optional symbol drawn at the anchor (e.g. ``"✕"``); empty for a label only.

    Returns:
        An ``ipyleaflet.DivIcon`` anchored at the position.
    """
    from ipyleaflet import DivIcon

    shadow = "text-shadow:0 0 3px #000,0 0 2px #000"
    sym = (f'<div style="position:absolute;left:-8px;top:-14px;color:{color};font:bold 20px sans-serif;{shadow}">'
           f"{symbol}</div>") if symbol else ""
    lab = (f'<div style="position:absolute;left:10px;top:-22px;color:{color};font:bold 13px sans-serif;'
           f'white-space:nowrap;{shadow}">{text}</div>')
    return DivIcon(html=sym + lab, icon_size=[0, 0], icon_anchor=[0, 0], class_name="geosplat-label")


def opacity_sliders(m: object, overlay: object | None = None) -> object:
    """Opacity sliders for the basemap (all base layers) and, optionally, an overlay.

    The sliders are linked in the browser (no kernel round trip).

    Args:
        m: ``ipyleaflet.Map``.
        overlay: Optional layer (e.g. the orthophoto) with its own slider.

    Returns:
        An ``ipywidgets.HBox`` with the sliders.
    """
    import ipywidgets as widgets

    sliders = []
    if overlay is not None:
        ov = widgets.FloatSlider(value=overlay.opacity, min=0, max=1, step=0.05, description="orthophoto")
        widgets.jslink((ov, "value"), (overlay, "opacity"))
        sliders.append(ov)
    base = widgets.FloatSlider(value=1.0, min=0, max=1, step=0.05, description="basemap")
    for layer in m.layers:
        if getattr(layer, "base", False):
            widgets.jslink((base, "value"), (layer, "opacity"))
    sliders.append(base)
    return widgets.HBox(sliders)


def photo_locations(image_dir: str | Path) -> list[tuple[Path, float, float]]:
    """Positions of the photos in a folder (EXIF, sidecars or external files, see :mod:`geosplat.positions`).

    Args:
        image_dir: Folder with images.

    Returns:
        List of ``(path, latitude, longitude)`` for photos with a known position.
    """
    positions = photo_positions(image_dir)
    out = []
    for p in list_images(image_dir):
        pos = position_of(p, positions)
        if pos is not None:
            out.append((p, pos.lat, pos.lon))
    return out


def _offset(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    """Move a lat/lon position by a small metric offset.

    Args:
        lat: Latitude (degrees).
        lon: Longitude (degrees).
        east_m: Eastward offset in meters.
        north_m: Northward offset in meters.

    Returns:
        New ``(lat, lon)``.
    """
    return lat + north_m / 111_320.0, lon + east_m / (111_320.0 * np.cos(np.radians(lat)))


def photo_kind(path: Path) -> tuple[str, float]:
    """Classify a photo's viewing direction from its metadata.

    Args:
        path: Photo path.

    Returns:
        ``(kind, yaw_deg)`` where kind is ``"nadir"`` (pitch <= -75 deg),
        ``"oblique"`` (known heading, not nadir) or ``"unknown"`` (no metadata).
    """
    pos = position_of(path, photo_positions(Path(path).parent))
    o = (pos.yaw, pos.pitch) if pos is not None and pos.yaw is not None else read_orientation(path)
    if o is None:
        return "unknown", 0.0
    yaw, pitch = o
    return ("nadir" if pitch is not None and pitch <= -75 else "oblique"), yaw


def photo_icon(kind: str, yaw: float, color: str = "#ff2fb3", size: int = 14) -> object:
    """Marker icon for a photo.

    * ``"nadir"``: square.
    * ``"oblique"``: camera symbol, a small circle at the photo position with a
      triangle (view cone) whose tip touches the circle and whose wide side faces
      the viewing direction.
    * ``"unknown"``: circle.

    Args:
        kind: ``"nadir"``, ``"oblique"`` or ``"unknown"``.
        yaw: Heading in degrees, clockwise from north (orients the view cone).
        color: Fill color.
        size: Size in pixels of the square / circle (the camera symbol is larger).

    Returns:
        An ``ipyleaflet.DivIcon`` centered on the photo position.
    """
    from ipyleaflet import DivIcon

    stroke = 'stroke="#222" stroke-width="1.5"'
    if kind == "oblique":
        r = size / 4                             # circle radius at the photo position
        length, half_width = size * 0.95, size * 0.55
        box = int(np.ceil(2 * np.hypot(r + length, half_width) + 6))   # room for the cone in any direction
        c = box / 2
        cone = (f"{c},{c - r} {c - half_width},{c - r - length} {c + half_width},{c - r - length}")
        shape = (f'<g transform="rotate({yaw:.1f} {c} {c})">'
                 f'<polygon points="{cone}" fill="{color}" fill-opacity="0.85" {stroke}/></g>'
                 f'<circle cx="{c}" cy="{c}" r="{r}" fill="{color}" {stroke}/>')
    else:
        box = size
        c = box / 2
        if kind == "nadir":
            shape = f'<rect x="2" y="2" width="{size - 4}" height="{size - 4}" fill="{color}" {stroke}/>'
        else:
            shape = f'<circle cx="{c}" cy="{c}" r="{c - 2}" fill="{color}" {stroke}/>'
    svg = f'<svg width="{box}" height="{box}" xmlns="http://www.w3.org/2000/svg">{shape}</svg>'
    return DivIcon(html=svg, icon_size=[box, box], icon_anchor=[c, c], class_name="geosplat-photo")


class PhotoSelector:
    """Map of photo positions to choose which photos to process.

    All photos start included. Tools (combine them freely):

    * **Mode** (Include / Exclude) + draw a polygon or rectangle: photos inside
      are included or excluded. The shape is removed once applied.
    * **Click a photo** to toggle it.
    * **Include all / Exclude all / Invert** buttons.

    Included photos are pink, excluded ones grey. Marker shape shows the viewing
    direction from metadata: square = near nadir, camera symbol = oblique,
    circle = unknown. Use :attr:`selected` or :meth:`save`, or pass
    ``SfMConfig(image_list=[p.name for p in selector.selected])``.

    Args:
        image_dir: Folder with GPS-tagged photos.
        basemap: Basemap name (see :data:`BASEMAPS`).
        height: Map height (CSS).
        selection_file: Optional text file (one name per line) to load a previous
            selection from (if it exists) and save to on every change.
    """

    INCLUDED = "#ff2fb3"
    EXCLUDED = "#9a9a9a"

    def __init__(
        self,
        image_dir: str | Path,
        basemap: str = "google_hybrid",
        height: str = "520px",
        selection_file: str | Path | None = None,
    ) -> None:
        """Build the map widget.

        Args:
            image_dir: Folder with GPS-tagged photos.
            basemap: Basemap name.
            height: Map height (CSS).
            selection_file: Optional selection file to load/save.

        Raises:
            ValueError: If no photo has GPS.
        """
        import ipywidgets as widgets
        from ipyleaflet import DrawControl, LayerGroup, Map, Marker, ScaleControl

        self.photos = photo_locations(image_dir)
        if not self.photos:
            raise ValueError("No GPS-tagged photos to show.")
        self.selection_file = Path(selection_file) if selection_file else None
        self.included = np.ones(len(self.photos), dtype=bool)
        if self.selection_file and self.selection_file.exists():
            names = set(self.selection_file.read_text().split())
            self.included = np.array([p.name in names for p, _, _ in self.photos])

        lats = np.array([p[1] for p in self.photos])
        lons = np.array([p[2] for p in self.photos])
        self.map = Map(center=(float(lats.mean()), float(lons.mean())), zoom=17, max_zoom=MAX_ZOOM, basemap=_basemap(basemap),
                       scroll_wheel_zoom=True, layout=widgets.Layout(height=height))
        self.map.add(ScaleControl(position="bottomleft"))
        _add_other_google_layers(self.map, basemap)

        self.kinds = [photo_kind(p) for p, _, _ in self.photos]
        self.markers = []
        for i, ((p, la, lo), (kind, yaw)) in enumerate(zip(self.photos, self.kinds)):
            m = Marker(location=(float(la), float(lo)), icon=photo_icon(kind, yaw), draggable=False, title=p.name)
            m.on_click(lambda i=i, **_: self._toggle(i))
            self.markers.append(m)
        self.map.add(LayerGroup(layers=self.markers, name="photos"))
        self.map.fit_bounds([[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]])

        style = {"shapeOptions": {"color": "#ffcc00", "fillOpacity": 0.15}}
        self.draw = DrawControl(polygon=style, rectangle=style, polyline={}, circlemarker={}, circle={},
                                marker={}, edit=False, remove=False)
        self.draw.on_draw(self._on_draw)
        self.map.add(self.draw)

        self.mode = widgets.ToggleButtons(options=["Include", "Exclude"], value="Exclude",
                                          tooltips=["Shapes include photos", "Shapes exclude photos"],
                                          description="Shapes:", style={"button_width": "80px"})
        buttons = []
        for text, action in (("Include all", lambda: self._set(np.ones_like(self.included))),
                             ("Exclude all", lambda: self._set(np.zeros_like(self.included))),
                             ("Invert", lambda: self._set(~self.included))):
            btn = widgets.Button(description=text, layout=widgets.Layout(width="110px"))
            btn.on_click(lambda _b, action=action: action())
            buttons.append(btn)
        self.label = widgets.HTML()
        help_text = widgets.HTML(
            "Draw a polygon/rectangle (toolbar, left) to <b>include</b> or <b>exclude</b> the photos inside "
            "(choose the mode); click a photo to toggle it. Pink = included, grey = excluded.")
        self.widget = widgets.VBox([help_text, widgets.HBox([self.mode, *buttons]), self.map,
                                    widgets.HBox([self.label, opacity_sliders(self.map)])])
        self._update()

    def _on_draw(self, target: object, action: str, geo_json: dict) -> None:
        """Apply a newly drawn shape (include or exclude the photos inside), then remove it.

        Args:
            target: The draw control.
            action: ``"created"`` (other actions are ignored).
            geo_json: The drawn feature.
        """
        import shapely
        from shapely.geometry import shape

        if action != "created" or not geo_json.get("geometry"):
            return
        geom = shape(geo_json["geometry"])
        lons = np.array([p[2] for p in self.photos])
        lats = np.array([p[1] for p in self.photos])
        inside = shapely.contains_xy(geom, lons, lats)
        new = self.included.copy()
        new[inside] = self.mode.value == "Include"
        self.draw.clear()
        self._set(new)

    def _toggle(self, i: int) -> None:
        """Toggle one photo.

        Args:
            i: Photo index.
        """
        new = self.included.copy()
        new[i] = not new[i]
        self._set(new)

    def _set(self, included: np.ndarray) -> None:
        """Replace the selection and refresh the map.

        Args:
            included: Boolean mask ``[num_photos]``.
        """
        self.included = np.asarray(included, dtype=bool)
        self._update()

    def _update(self) -> None:
        """Recolor markers, refresh the count and persist the selection."""
        for m, (kind, yaw), ok in zip(self.markers, self.kinds, self.included):
            color = self.INCLUDED if ok else self.EXCLUDED
            if getattr(m, "_color", None) != color:
                m.icon = photo_icon(kind, yaw, color)
                m._color = color
        n = int(self.included.sum())
        self.label.value = f"<b>{n}</b> of {len(self.photos)} photos included"
        if self.selection_file:
            self.save(self.selection_file)

    @property
    def selected(self) -> list[Path]:
        """Included photos.

        Returns:
            Paths of the included photos.
        """
        return [p for (p, _, _), ok in zip(self.photos, self.included) if ok]

    @property
    def all_selected(self) -> bool:
        """Whether every photo is included.

        Returns:
            ``True`` if nothing is excluded.
        """
        return bool(self.included.all())

    def save(self, path: str | Path) -> Path:
        """Write the included photo names, one per line (usable as ``--image-list``).

        Args:
            path: Output text file.

        Returns:
            The written path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(p.name for p in self.selected) + "\n")
        return path

    def _ipython_display_(self) -> None:
        """Display the widget in a notebook."""
        from IPython.display import display

        display(self.widget)


def _png_data_url(rgba: np.ndarray) -> str:
    """Encode an RGBA array as a PNG data URL.

    Args:
        rgba: ``uint8`` array ``[H, W, 4]``.

    Returns:
        ``data:image/png;base64,...`` string.
    """
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def ortho_overlay(tif_path: str | Path, max_size: int = 2048, opacity: float = 1.0) -> tuple[object, tuple]:
    """Reproject a GeoTIFF to Web Mercator and wrap it as an ipyleaflet image overlay.

    Args:
        tif_path: RGB/RGBA GeoTIFF with a CRS.
        max_size: Longest side of the overlay in pixels.
        opacity: Overlay opacity.

    Returns:
        Tuple ``(ImageOverlay, ((south, west), (north, east)))``.
    """
    import rasterio
    from ipyleaflet import ImageOverlay
    from pyproj import Transformer
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    with rasterio.open(tif_path) as src:
        transform, width, height_px = calculate_default_transform(src.crs, "EPSG:3857", src.width, src.height, *src.bounds)
        scale = max(width, height_px) / max_size
        if scale > 1:
            transform = transform * transform.scale(scale, scale)
            width, height_px = int(np.ceil(width / scale)), int(np.ceil(height_px / scale))
        bands = min(src.count, 4)
        dst = np.zeros((bands, height_px, width), dtype=np.uint8)
        for b in range(bands):
            reproject(rasterio.band(src, b + 1), dst[b], src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs="EPSG:3857", resampling=Resampling.bilinear)
    rgba = np.moveaxis(dst, 0, -1)
    if bands == 3:
        rgba = np.concatenate([rgba, np.where(rgba.any(-1, keepdims=True), 255, 0).astype(np.uint8)], -1)
    to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    west, north = to_ll.transform(transform.c, transform.f)
    east, south = to_ll.transform(transform.c + transform.a * width, transform.f + transform.e * height_px)
    bounds = ((south, west), (north, east))
    return ImageOverlay(url=_png_data_url(rgba), bounds=bounds, opacity=opacity, name=Path(tif_path).name), bounds


def ortho_map(
    tif_path: str | Path,
    basemap: str = "google_satellite",
    max_size: int = 2048,
    gcps: dict[str, np.ndarray] | None = None,
    height: str = "560px",
    opacity: float = 1.0,
) -> object:
    """Show a georeferenced orthophoto GeoTIFF over a satellite basemap.

    The raster is reprojected to Web Mercator (EPSG:3857) at display
    resolution, so it lines up exactly with the basemap.

    Args:
        tif_path: RGB/RGBA GeoTIFF with a CRS.
        basemap: Basemap name (see :data:`BASEMAPS`).
        max_size: Longest side of the displayed overlay in pixels.
        gcps: Optional ``{name: [x, y, z]}`` in the raster's CRS, drawn as markers.
        height: Map height (CSS).
        opacity: Initial overlay opacity.

    Returns:
        An ``ipywidgets.VBox`` with the map and opacity sliders (orthophoto, basemap).
    """
    import ipywidgets as widgets
    import rasterio
    from ipyleaflet import CircleMarker, Map, ScaleControl
    from pyproj import Transformer

    overlay, ((south, west), (north, east)) = ortho_overlay(tif_path, max_size, opacity)
    m = Map(center=((north + south) / 2, (east + west) / 2), zoom=18, max_zoom=MAX_ZOOM, basemap=_basemap(basemap),
            scroll_wheel_zoom=True, layout=widgets.Layout(height=height))
    m.add(overlay)
    m.add(ScaleControl(position="bottomleft"))
    _add_other_google_layers(m, basemap)
    if gcps:
        with rasterio.open(tif_path) as src:
            t = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        from ipyleaflet import Marker

        for name, xyz in gcps.items():
            lon, lat = t.transform(xyz[0], xyz[1])
            m.add(CircleMarker(location=(lat, lon), radius=6, color="red", fill_opacity=0, weight=2, title=name))
            m.add(Marker(location=(lat, lon), icon=label_icon(name), draggable=False, title=name))
    m.fit_bounds([[south, west], [north, east]])
    return widgets.VBox([m, opacity_sliders(m, overlay)])
