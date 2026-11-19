"""Ground control points: pick them on the orthophoto and refine the georeference.

Workflow (notebook):

1. :class:`GCPPicker` shows the orthophoto over Google Satellite with the GCP
   coordinates as red circles. Select a GCP name, click where its target is in
   the orthophoto; drag markers to move them, delete or clear as needed.
2. :func:`refine_georeference` turns the picks into 3D points (height from the
   DSM) and fits a 7-parameter similarity (>= 3 points) or a translation
   (1-2 points) mapping the GPS-based georeference onto the GCPs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geo import GeoReference, umeyama


class GCPPicker:
    """Interactive map to locate ground control points on an orthophoto.

    Args:
        ortho_tif: Georeferenced orthophoto GeoTIFF to click on.
        gcps: ``{name: [x, y, z]}`` in the orthophoto's CRS.
        picks_file: Optional JSON file where picks are saved on every change
            (and loaded from, if it exists).
        height: Map height (CSS).
    """

    def __init__(
        self,
        ortho_tif: str | Path,
        gcps: dict[str, np.ndarray],
        picks_file: str | Path | None = None,
        height: str = "600px",
    ) -> None:
        """Build the picking widget.

        Args:
            ortho_tif: Orthophoto GeoTIFF.
            gcps: Ground control points in the orthophoto's CRS.
            picks_file: Optional JSON file for persisting picks.
            height: Map height (CSS).
        """
        import ipywidgets as widgets
        import rasterio
        from ipyleaflet import CircleMarker, LayerGroup, Map, Marker, ScaleControl
        from pyproj import Transformer

        from .mapview import _add_other_google_layers, _basemap, label_icon, ortho_overlay

        self.gcps = dict(gcps)
        self.names = list(self.gcps)
        self.picks_file = Path(picks_file) if picks_file else None
        with rasterio.open(ortho_tif) as src:
            self.crs = src.crs
        to_ll = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

        overlay, ((south, west), (north, east)) = ortho_overlay(ortho_tif, max_size=4096)
        self.map = Map(center=((north + south) / 2, (east + west) / 2), zoom=19, max_zoom=24,
                       basemap=_basemap("google_satellite"), scroll_wheel_zoom=True,
                       layout=widgets.Layout(height=height))
        self.map.add(overlay)
        self.map.add(ScaleControl(position="bottomleft"))
        _add_other_google_layers(self.map, "google_satellite")
        refs = []
        for name, xyz in self.gcps.items():
            lon, lat = to_ll.transform(xyz[0], xyz[1])
            refs.append(CircleMarker(location=(lat, lon), radius=7, color="red", fill_opacity=0, weight=2,
                                     title=f"{name} (coordinates in the CSV)"))
            refs.append(Marker(location=(lat, lon), icon=label_icon(name, "#ff3333"), draggable=False,
                               title=f"{name} (coordinates in the CSV)"))
        self.map.add(LayerGroup(layers=refs, name="GCP coordinates"))
        self.map.fit_bounds([[south, west], [north, east]])

        self.markers: dict[str, object] = {}
        self.name_select = widgets.Dropdown(options=self.names, description="Place:")
        delete_btn = widgets.Button(description="Delete selected", icon="trash")
        clear_btn = widgets.Button(description="Clear all", icon="times")
        delete_btn.on_click(lambda _: self._remove(self.name_select.value))
        clear_btn.on_click(lambda _: [self._remove(n) for n in list(self.markers)])
        self.table = widgets.HTML()
        help_text = widgets.HTML(
            "Choose a point, then <b>click its target on the orthophoto</b>. Drag a marker to move it. "
            "Red circles = where the CSV coordinates fall with the current (GPS) georeference; "
            "pink ✕ = your picks. "
            "Pick at least 3 points for a full fit (1-2 points: shift only)."
        )
        self.map.on_interaction(self._on_click)
        self.widget = widgets.VBox([help_text, widgets.HBox([self.name_select, delete_btn, clear_btn]),
                                    self.map, self.table])
        if self.picks_file and self.picks_file.exists():
            for name, (lat, lon) in json.loads(self.picks_file.read_text()).items():
                if name in self.gcps:
                    self._place(name, (lat, lon))
        self._refresh()

    def _marker_icon(self, name: str) -> object:
        """Pink ✕ with the GCP name, used for picked positions.

        Args:
            name: GCP name shown next to the ✕.

        Returns:
            An ``ipyleaflet.DivIcon``.
        """
        from .mapview import label_icon

        return label_icon(name, "#ff2fb3", symbol="✕")

    def _place(self, name: str, latlon: tuple[float, float]) -> None:
        """Place (or move) the marker of a GCP.

        Args:
            name: GCP name.
            latlon: ``(lat, lon)`` position.
        """
        from ipyleaflet import Marker

        if name in self.markers:
            self.markers[name].location = latlon
            return
        marker = Marker(location=latlon, draggable=True, icon=self._marker_icon(name), title=name)
        marker.observe(lambda _change: self._refresh(), "location")
        self.markers[name] = marker
        self.map.add(marker)

    def _remove(self, name: str | None) -> None:
        """Delete the marker of a GCP.

        Args:
            name: GCP name.
        """
        if name in self.markers:
            self.map.remove(self.markers.pop(name))
            self._refresh()

    def _on_click(self, **event: object) -> None:
        """Handle map clicks: place the selected GCP and advance to the next unplaced one.

        Args:
            **event: ipyleaflet interaction event (``type``, ``coordinates``).
        """
        if event.get("type") != "click" or self.name_select.value is None:
            return
        lat, lon = event["coordinates"]  # type: ignore[misc]
        self._place(self.name_select.value, (float(lat), float(lon)))
        remaining = [n for n in self.names if n not in self.markers]
        if remaining:
            self.name_select.value = remaining[0]
        self._refresh()

    def _refresh(self) -> None:
        """Update the table of picks and persist them."""
        rows = "".join(
            f"<tr><td>{n}</td><td>{'&#10003;' if n in self.markers else ''}</td>"
            f"<td>{self.gcps[n][0]:.3f}</td><td>{self.gcps[n][1]:.3f}</td><td>{self.gcps[n][2]:.3f}</td></tr>"
            for n in self.names
        )
        self.table.value = (f"<b>{len(self.markers)}/{len(self.names)} points placed</b>"
                            f"<table><tr><th>name</th><th>placed</th><th>x</th><th>y</th><th>z</th></tr>{rows}</table>")
        if self.picks_file:
            self.picks_file.parent.mkdir(parents=True, exist_ok=True)
            self.picks_file.write_text(json.dumps(self.picks, indent=2))

    @property
    def picks(self) -> dict[str, tuple[float, float]]:
        """Placed points.

        Returns:
            ``{name: (lat, lon)}``.
        """
        return {n: tuple(m.location) for n, m in self.markers.items()}

    def _ipython_display_(self) -> None:
        """Display the widget in a notebook."""
        from IPython.display import display

        display(self.widget)


def load_picks(path: str | Path) -> dict[str, tuple[float, float]]:
    """Load picks saved by :class:`GCPPicker`.

    Args:
        path: JSON file.

    Returns:
        ``{name: (lat, lon)}``.
    """
    return {k: tuple(v) for k, v in json.loads(Path(path).read_text()).items()}


def refine_georeference(
    geo: GeoReference,
    picks: dict[str, tuple[float, float]],
    gcps: dict[str, np.ndarray],
    dsm_tif: str | Path,
) -> tuple[GeoReference, dict[str, object]]:
    """Correct a GPS-based georeference with ground control points picked on the orthophoto.

    Each pick (lat/lon on the current orthophoto) becomes a 3D point in the
    current georeference, with its height sampled from ``dsm_tif``. The
    correction maps these points onto the GCP coordinates: a 7-parameter
    similarity for 3+ points, a 3D translation for 1-2 points.

    Args:
        geo: Current georeference (from photo GPS).
        picks: ``{name: (lat, lon)}`` from :class:`GCPPicker`.
        gcps: ``{name: [x, y, z]}`` in ``geo.crs``.
        dsm_tif: DSM GeoTIFF made with ``geo`` (``ortho/dsm.tif`` or ``fast_orto/dsm_sparse.tif``).

    Returns:
        Tuple ``(new_geo, report)``; ``report`` has per-point residuals (m), RMSE,
        the method and the scale change.

    Raises:
        ValueError: If no pick matches a GCP name.
    """
    import rasterio
    from pyproj import Transformer

    names = [n for n in picks if n in gcps]
    if not names:
        raise ValueError("No picked point matches a GCP name.")
    to_crs = Transformer.from_crs("EPSG:4326", geo.crs, always_xy=True)
    east, north = to_crs.transform([picks[n][1] for n in names], [picks[n][0] for n in names])
    with rasterio.open(dsm_tif) as dsm:
        heights = np.array([v[0] for v in dsm.sample(list(zip(east, north)))], dtype=np.float64)
        nodata = dsm.nodata
    if nodata is not None and np.any(heights == nodata):
        valid = heights[heights != nodata]
        heights[heights == nodata] = np.median(valid) if len(valid) else 0.0
    src = np.stack([east, north, heights], 1) - geo.origin   # where the targets are now (local)
    dst = np.stack([gcps[n] for n in names]) - geo.origin   # where they should be (local)

    if len(names) >= 3:
        C, method = umeyama(src, dst), "similarity (7 parameters)"
    else:
        C, method = np.eye(4), "translation"
        C[:3, 3] = (dst - src).mean(0)
    fitted = src @ C[:3, :3].T + C[:3, 3]
    res = fitted - dst
    report: dict[str, object] = {
        "method": method,
        "points": len(names),
        "residuals_m": {n: [round(float(v), 3) for v in r] for n, r in zip(names, res)},
        "rmse_xy_m": float(np.sqrt((res[:, :2] ** 2).sum(1).mean())),
        "rmse_z_m": float(np.sqrt((res[:, 2] ** 2).mean())),
        "shift_before_m": {n: [round(float(v), 3) for v in d] for n, d in zip(names, dst - src)},
        "scale_change": float(np.cbrt(np.linalg.det(C[:3, :3]))),
    }
    new_geo = GeoReference(
        geo.crs, geo.origin.copy(), C @ geo.scene_to_geo, report["rmse_xy_m"], geo.num_cameras,  # type: ignore[arg-type]
        {**geo.extra, "georeferencing": f"GCPs ({method})", "gcp_report": report},
    )
    return new_geo, report
