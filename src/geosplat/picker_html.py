"""Stand-alone HTML page to locate ground control points on an orthophoto.

:func:`save_picker_html` writes ``locate_targets.html`` (plus a folder of orthophoto
tiles). Open it in any browser, no Python needed:

1. Choose a point in the list, click its target on the orthophoto (drag to adjust).
2. Press **Save located_targets.json** and save it next to the page (``<output>/targets/located_targets.json``).
3. Run the same command again: the picks correct the georeference of every
   product (SfM and the trained splat are reused).

Each pick stores latitude, longitude and the DSM height at that spot, plus
the georeference of the orthophoto, so it stays valid when products are
regenerated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geo import GeoReference


def _to_3857(src_path: Path, max_size: int, resampling: str = "bilinear") -> tuple[np.ndarray, object, float]:
    """Reproject a raster to Web Mercator.

    Args:
        src_path: GeoTIFF with a CRS.
        max_size: Longest side of the result in pixels.
        resampling: Rasterio resampling name.

    Returns:
        Tuple ``(array [bands, H, W], affine transform, nodata)``.
    """
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(src.crs, "EPSG:3857", src.width, src.height,
                                                               *src.bounds)
        scale = max(width, height) / max_size
        if scale > 1:
            transform = transform * transform.scale(scale, scale)
            width, height = int(np.ceil(width / scale)), int(np.ceil(height / scale))
        nodata = src.nodata if src.nodata is not None else 0
        dst = np.full((src.count, height, width), nodata, dtype=src.dtypes[0])
        for b in range(src.count):
            reproject(rasterio.band(src, b + 1), dst[b], src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs="EPSG:3857", src_nodata=src.nodata,
                      dst_nodata=src.nodata,           # None for RGBA: transparency comes from the alpha band
                      resampling=getattr(Resampling, resampling))
    return dst, transform, nodata


def _bounds_ll(transform: object, col0: int, row0: int, w: int, h: int) -> list[list[float]]:
    """Lat/lon bounds of a pixel window of a Web Mercator raster.

    Args:
        transform: Affine transform of the raster (EPSG:3857).
        col0: First column.
        row0: First row.
        w: Window width.
        h: Window height.

    Returns:
        ``[[south, west], [north, east]]``.
    """
    from pyproj import Transformer

    to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    west, north = to_ll.transform(*(transform @ (col0, row0)))
    east, south = to_ll.transform(*(transform @ (col0 + w, row0 + h)))
    return [[south, west], [north, east]]


def save_picker_html(
    ortho_tif: str | Path,
    dsm_tif: str | Path,
    gcps: dict[str, np.ndarray],
    geo: GeoReference,
    html_path: str | Path,
    picks: dict[str, list[float]] | None = None,
    scene_transform: np.ndarray | None = None,
    max_size: int = 16_000,
    tile: int = 2048,
    dsm_size: int = 300,
) -> Path:
    """Write the stand-alone GCP picking page.

    Args:
        ortho_tif: Georeferenced orthophoto to click on.
        dsm_tif: DSM made with the same georeference (heights of the picks).
        gcps: ``{name: [x, y, z]}`` in ``geo.crs``.
        geo: Georeference the orthophoto was made with (stored in the saved picks).
        html_path: Output page (tiles go to ``<stem>_files/`` next to it).
        picks: Picks to show at start, ``{name: [lat, lon, ...]}``.
        scene_transform: COLMAP-to-scene transform of the run, stored in the picks
            to detect a changed SfM reconstruction.
        max_size: Longest side of the orthophoto in the page (pixels).
        tile: Tile size in pixels.
        dsm_size: Longest side of the DSM grid embedded for heights.

    Returns:
        Path of the page.
    """
    from PIL import Image
    from pyproj import Transformer

    html_path = Path(html_path)
    tiles_dir = html_path.parent / f"{html_path.stem}_files"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    for old in tiles_dir.glob("*.webp"):
        old.unlink()

    rgba, transform, _ = _to_3857(Path(ortho_tif), max_size)
    rgba = np.moveaxis(rgba[:4], 0, -1)
    if rgba.shape[-1] == 3:
        rgba = np.concatenate([rgba, np.where(rgba.any(-1, keepdims=True), 255, 0).astype(np.uint8)], -1)
    H, W = rgba.shape[:2]
    tiles = []
    for r in range(0, H, tile):
        for c in range(0, W, tile):
            part = rgba[r : r + tile, c : c + tile]
            if not part[..., 3].any():
                continue
            name = f"{r // tile}_{c // tile}.webp"
            Image.fromarray(part, "RGBA").save(tiles_dir / name, quality=88, method=4)
            tiles.append({"url": f"{tiles_dir.name}/{name}",
                          "bounds": _bounds_ll(transform, c, r, part.shape[1], part.shape[0])})
    full_bounds = _bounds_ll(transform, 0, 0, W, H)

    dsm, dt, nodata = _to_3857(Path(dsm_tif), dsm_size, "nearest")
    z = dsm[0].astype(np.float64)
    z[(z == nodata) | ~np.isfinite(z)] = np.nan
    valid = z[np.isfinite(z)]
    fill = float(np.median(valid)) if len(valid) else 0.0
    dsm_info = {"west": dt.c, "north": dt.f, "dx": dt.a, "dy": dt.e, "w": z.shape[1], "h": z.shape[0],
                "fill": round(fill, 3), "z": [None if not np.isfinite(v) else round(float(v), 3) for v in z.ravel()]}

    to_ll = Transformer.from_crs(geo.crs, "EPSG:4326", always_xy=True)
    points = []
    for name, xyz in gcps.items():
        lon, lat = to_ll.transform(float(xyz[0]), float(xyz[1]))
        points.append({"name": name, "lat": lat, "lon": lon, "x": float(xyz[0]), "y": float(xyz[1]),
                       "z": None if not np.isfinite(xyz[2]) else float(xyz[2])})

    ref = {"crs": geo.crs, "origin": geo.origin.tolist(), "scene_to_geo": geo.scene_to_geo.tolist(),
           "rmse_m": geo.rmse, "num_cameras": geo.num_cameras}
    data = {
        "tiles": tiles, "bounds": full_bounds, "gcps": points, "dsm": dsm_info,
        "picks": picks or {}, "geo": ref, "ortho": Path(ortho_tif).name,
        "scene_transform": None if scene_transform is None else np.asarray(scene_transform).tolist(),
        "save_to": str((html_path.parent / "located_targets.json").resolve()), "key": str(html_path.resolve()),
    }
    html_path.write_text(_PAGE.replace("/*DATA*/null", json.dumps(data)), encoding="utf-8")
    return html_path


def picks_heights(data: dict) -> dict[str, tuple[float, ...]]:
    """Picks of a saved ``located_targets.json`` (page or notebook format).

    Args:
        data: Parsed JSON: ``{"picks": {name: [lat, lon, h]}, "geo": ...}`` from
            the page, or ``{name: [lat, lon]}`` from the notebook picker.

    Returns:
        ``{name: (lat, lon)}`` or ``{name: (lat, lon, height)}``.
    """
    picks = data.get("picks", data) if isinstance(data.get("picks"), dict) else data
    return {k: tuple(float(x) for x in v) for k, v in picks.items() if isinstance(v, (list, tuple))}


_PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>geosplat · locate targets</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html,body{margin:0;height:100%;font:14px sans-serif}
 #wrap{display:flex;height:100%}
 #side{width:300px;padding:10px;overflow:auto;box-sizing:border-box;border-right:1px solid #ccc}
 #map{flex:1}
 .pt{padding:5px 6px;margin:2px 0;border-radius:4px;cursor:pointer;display:flex;justify-content:space-between}
 .pt.sel{background:#ffd6f0;font-weight:bold}
 .pt .del{color:#a00;cursor:pointer;padding:0 4px}
 button{font-size:15px;padding:8px 10px;margin:4px 0;width:100%;cursor:pointer}
 #save{background:#ff2fb3;color:#fff;border:0;border-radius:4px;font-weight:bold}
 .lab{color:var(--c);font:bold 13px sans-serif;white-space:nowrap;text-shadow:0 0 3px #000,0 0 2px #000}
 .sym{position:absolute;left:-8px;top:-14px;font:bold 20px sans-serif}
 .txt{position:absolute;left:10px;top:-22px}
 small{color:#555} label{display:block;margin-top:6px} code{word-break:break-all}
</style></head><body>
<div id="wrap"><div id="side">
 <b>Locate the targets</b><br>
 <small>1. Choose a point. 2. Click its target on the orthophoto (drag to adjust).
 3. <b>Save</b> as <code>located_targets.json</code> in the folder of this page, then run the same command again.
 Red circles: target coordinates with the current georeference. Pink ✕: your picks. 3+ points: full fit.</small>
 <div id="list"></div>
 <button id="save">Save located_targets.json</button>
 <button id="clear">Clear all</button>
 <small id="status"></small>
 <label>Orthophoto opacity <input id="oop" type="range" min="0" max="1" step="0.05" value="1"></label>
 <label>Background opacity <input id="bop" type="range" min="0" max="1" step="0.05" value="1"></label>
 <small>Save to: <code id="saveto"></code></small>
</div><div id="map"></div></div>
<script>
const D = /*DATA*/null;
const KEY = "geosplat-picks:" + D.key;
const map = L.map("map", {maxZoom: 24});
const g = (l) => L.tileLayer("https://mt1.google.com/vt/lyrs=" + l + "&x={x}&y={y}&z={z}",
  {maxZoom: 24, maxNativeZoom: 21, attribution: "Google"});
const bases = {"Google Satellite": g("s"), "Google Hybrid": g("y")};
bases["Google Satellite"].addTo(map);
const ortho = L.layerGroup(D.tiles.map(t => L.imageOverlay(t.url, t.bounds))).addTo(map);
const refs = L.layerGroup().addTo(map);
L.control.layers(bases, {"Orthophoto": ortho, "Target coordinates": refs}).addTo(map);
L.control.scale({imperial: false}).addTo(map);
map.fitBounds(D.bounds);
const icon = (name, color, sym) => L.divIcon({className: "", iconSize: [0, 0], iconAnchor: [0, 0],
  html: '<div class="lab" style="--c:' + color + '">' + (sym ? '<div class="sym">' + sym + '</div>' : '') +
        '<div class="txt">' + name + '</div></div>'});
for (const p of D.gcps) {
  L.circleMarker([p.lat, p.lon], {radius: 7, color: "red", fill: false, weight: 2}).addTo(refs);
  L.marker([p.lat, p.lon], {icon: icon(p.name, "#ff3333"), interactive: false}).addTo(refs);
}
document.getElementById("saveto").textContent = D.save_to;
document.getElementById("oop").oninput = e => ortho.eachLayer(l => l.setOpacity(+e.target.value));
document.getElementById("bop").oninput = e => Object.values(bases).forEach(l => l.setOpacity(+e.target.value));

const names = D.gcps.map(p => p.name);
const markers = {};
let sel = names[0];
function height(lat, lon) {   // DSM height (bilinear, nodata-aware) at a WGS84 position
  const x = lon * 20037508.342789244 / 180;
  const y = Math.log(Math.tan((90 + lat) * Math.PI / 360)) * 6378137;
  const d = D.dsm, c = (x - d.west) / d.dx - 0.5, r = (y - d.north) / d.dy - 0.5;
  const c0 = Math.floor(c), r0 = Math.floor(r);
  let s = 0, w = 0;
  for (const [dc, dr] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
    const cc = c0 + dc, rr = r0 + dr;
    if (cc < 0 || rr < 0 || cc >= d.w || rr >= d.h) continue;
    const v = d.z[rr * d.w + cc];
    if (v === null) continue;
    const wt = (1 - Math.abs(c - cc)) * (1 - Math.abs(r - rr));
    s += v * wt; w += wt;
  }
  return w > 0 ? s / w : d.fill;
}
function place(name, ll) {
  if (markers[name]) { markers[name].setLatLng(ll); return; }
  const m = L.marker(ll, {draggable: true, icon: icon(name, "#ff2fb3", "✕")}).addTo(map);
  m.on("dragend", refresh);
  markers[name] = m;
}
function picks() {
  const out = {};
  for (const [n, m] of Object.entries(markers)) {
    const ll = m.getLatLng();
    out[n] = [ll.lat, ll.lng, height(ll.lat, ll.lng)];
  }
  return out;
}
function payload() {
  return JSON.stringify({picks: picks(), geo: D.geo, scene_transform: D.scene_transform,
                         ortho: D.ortho, saved: new Date().toISOString()}, null, 1);
}
function refresh() {
  const list = document.getElementById("list");
  list.innerHTML = "";
  for (const n of names) {
    const div = document.createElement("div");
    div.className = "pt" + (n === sel ? " sel" : "");
    div.innerHTML = "<span>" + (markers[n] ? "✔ " : "○ ") + n + "</span>" + (markers[n] ? '<span class="del" title="delete">✖</span>' : "");
    div.onclick = (e) => {
      if (e.target.classList.contains("del")) { map.removeLayer(markers[n]); delete markers[n]; }
      else { sel = n; if (markers[n]) map.panTo(markers[n].getLatLng()); }
      refresh();
    };
    list.appendChild(div);
  }
  localStorage.setItem(KEY, payload());
  document.getElementById("status").textContent = Object.keys(markers).length + "/" + names.length + " points placed";
}
map.on("click", e => {
  if (!sel) return;
  place(sel, e.latlng);
  const left = names.filter(n => !markers[n]);
  if (left.length) sel = left[0];
  refresh();
});
document.getElementById("clear").onclick = () => {
  if (!confirm("Delete all picks?")) return;
  for (const n of Object.keys(markers)) { map.removeLayer(markers[n]); delete markers[n]; }
  refresh();
};
document.getElementById("save").onclick = async () => {
  const text = payload();
  if (window.showSaveFilePicker) {
    try {
      const h = await showSaveFilePicker({suggestedName: "located_targets.json",
        types: [{description: "JSON", accept: {"application/json": [".json"]}}]});
      const w = await h.createWritable(); await w.write(text); await w.close();
      document.getElementById("status").textContent = "Saved " + h.name + ". Run the same command again.";
      return;
    } catch (err) { if (err.name === "AbortError") return; }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], {type: "application/json"}));
  a.download = "located_targets.json"; a.click();
  document.getElementById("status").textContent = "Downloaded located_targets.json: move it to " + D.save_to;
};
// Start from this browser's last session of this page, else from the saved picks.
let start = D.picks;
try { const s = JSON.parse(localStorage.getItem(KEY) || "null"); if (s && JSON.stringify(s.geo) === JSON.stringify(D.geo)) start = s.picks; } catch (e) {}
for (const [n, v] of Object.entries(start || {})) if (names.includes(n)) place(n, [v[0], v[1]]);
const left0 = names.filter(n => !markers[n]); sel = left0.length ? left0[0] : names[0];
refresh();
</script></body></html>
"""
