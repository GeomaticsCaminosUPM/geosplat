"""Interactive Gaussian splat / point cloud viewer for notebooks (Jupyter / Colab) and standalone HTML.

The splat is converted to the compact ``.splat`` format (32 bytes per Gaussian,
most important Gaussians first) and embedded in a small HTML page that runs the
open-source WebGL viewer `@mkkellogg/gaussian-splats-3d
<https://github.com/mkkellogg/GaussianSplats3D>`_ (three.js), loaded from a CDN.

The page has two views, **Splats** and **Point cloud** (the dense points, one
per solid Gaussian, with a point size slider), and two color modes:

* **RGB**: the reconstructed colors.
* **Classes**, blended over the RGB colors with an opacity slider, in two tabs:

  * **Trees**: bare ground / ground vegetation / trees / structures (see
    :func:`geosplat.classify.classify_scene`): an object standing on the
    ground is a structure only with a high enough structure score, learned
    from all the properties of its Gaussians (see
    :func:`geosplat.classify.classify_scene`), otherwise a tree; ground is
    split by greenness.
  * **Objects**: one color per object standing on the ground (each tree,
    building, wall...), see :func:`geosplat.classify.segment_objects`.

  Points take the class of their Gaussian. Only the reconstructed area is
  classified; the far background keeps its colors.

Controls: left-drag orbit, right-drag pan, wheel zoom.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
import torch

from .scene import Camera
from .sh import sh_to_rgb

THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js"
VIEWER_URL = "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.6/build/gaussian-splats-3d.module.js"
GREENNESS_RANGE = 0.5   # greenness is stored clipped to [-0.5, 0.5]


@torch.no_grad()
def splat_bytes(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor], max_splats: int | None = 400_000,
    return_order: bool = False,
) -> bytes | tuple[bytes, np.ndarray]:
    """Encode Gaussians in the ``.splat`` format used by web viewers.

    Per Gaussian: position (3 x float32), scale (3 x float32), color RGBA
    (4 x uint8, alpha = opacity) and rotation (4 x uint8, ``wxyz`` mapped from
    ``[-1, 1]`` to ``[0, 255]``). Gaussians are sorted by ``opacity * volume``
    so truncation keeps the most visible ones.

    Args:
        params: Gaussian parameters (unconstrained form, as trained).
        max_splats: Keep at most this many Gaussians (``None`` keeps all).
        return_order: Also return the index of the Gaussian of each record.

    Returns:
        The binary ``.splat`` payload, plus the Gaussian indices with ``return_order``.
    """
    means = params["means"].detach().float()
    scales = torch.exp(params["scales"].detach().float())
    opacity = torch.sigmoid(params["opacities"].detach().float()).reshape(-1)
    importance = opacity * scales.prod(dim=1)
    order = torch.argsort(importance, descending=True)
    if max_splats is not None:
        order = order[:max_splats]
    rgb = sh_to_rgb(params["sh0"].detach().float()[order, 0]).clamp(0, 1)
    rgba = torch.cat([rgb, opacity[order, None]], dim=1)
    quats = torch.nn.functional.normalize(params["quats"].detach().float()[order], dim=1)

    n = len(order)
    buf = np.empty(n, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    buf["pos"] = means[order].cpu().numpy()
    buf["scale"] = scales[order].cpu().numpy()
    buf["rgba"] = (rgba * 255).round().clamp(0, 255).byte().cpu().numpy()
    buf["rot"] = (quats * 128 + 128).round().clamp(0, 255).byte().cpu().numpy()
    if return_order:
        return buf.tobytes(), order.cpu().numpy()
    return buf.tobytes()


def points_bytes(
    xyz: np.ndarray, rgb: np.ndarray, max_points: int | None = 400_000, index: np.ndarray | None = None
) -> tuple[bytes, bytes, float, np.ndarray | None]:
    """Encode a colored point cloud for the viewer.

    Args:
        xyz: Points ``[N, 3]`` in scene coordinates.
        rgb: Colors ``[N, 3]`` (``uint8``).
        max_points: Keep at most this many points (random subsample).
        index: Optional Gaussian index of each point (subsampled along).

    Returns:
        Tuple ``(positions float32 bytes, colors uint8 bytes, typical point
        spacing, kept Gaussian indices or None)``.
    """
    if max_points is not None and len(xyz) > max_points:
        keep = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
        index = None if index is None else index[keep]
    pos = np.ascontiguousarray(xyz, dtype=np.float32)
    col = np.ascontiguousarray(rgb, dtype=np.uint8)
    if len(pos) > 1:
        lo, hi = np.percentile(pos, [5, 95], axis=0)
        span = np.sort(hi - lo)[1:]                     # two largest extents (surface-like clouds)
        spacing = float(np.sqrt(span.prod() / len(pos)))
    else:
        spacing = 0.01
    return pos.tobytes(), col.tobytes(), max(spacing, 1e-6), index


def class_bytes(classes: dict[str, object], region: np.ndarray, index: np.ndarray) -> bytes:
    """Quantize the class inputs of some Gaussians to 4 bytes each.

    Args:
        classes: Output of :func:`geosplat.classify.classify_scene`.
        region: Per-Gaussian flag, True inside the reconstructed area.
        index: Gaussians to encode, in output order.

    Returns:
        ``len(index) * 4`` bytes: greenness, flags (1: region, 2: standing,
        4: foliage), object id (2 bytes, little-endian).
    """
    g = np.asarray(classes["greenness"])[index]
    g = (np.clip(g, -GREENNESS_RANGE, GREENNESS_RANGE) + GREENNESS_RANGE) / (2 * GREENNESS_RANGE)
    flags = (region[index] * 1 + np.asarray(classes["standing"])[index] * 2
             + np.asarray(classes["foliage"])[index] * 4)
    ids = np.clip(np.asarray(classes["ids"])[index], 0, 65535).astype(np.int64)
    cols = [np.round(g * 255), flags, ids & 255, ids >> 8]
    return np.ascontiguousarray(np.stack(cols, 1).astype(np.uint8)).tobytes()


def _initial_view(cameras: list[Camera] | None) -> tuple[list[float], list[float]]:
    """Choose an initial camera position and look-at point.

    Args:
        cameras: Training cameras in scene coordinates (centered scene, up = -Y).

    Returns:
        Tuple ``(camera_position, look_at)``.
    """
    if not cameras:
        return [0.0, -3.0, -6.0], [0.0, 0.0, 0.0]
    centers = np.stack([c.center for c in cameras])
    mean = centers.mean(0)
    dist = float(np.linalg.norm(centers, axis=1).mean())
    pos = mean * 1.3 if np.linalg.norm(mean) > 0.3 * dist else mean + np.array([0.0, -dist, -0.5 * dist])
    return pos.round(4).tolist(), [0.0, 0.0, 0.0]


def viewer_html(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    cameras: list[Camera] | None = None,
    max_splats: int | None = 400_000,
    background: str = "#101418",
    points: tuple[np.ndarray, ...] | None = None,
    max_points: int | None = 400_000,
    classes: bool = True,
    region: np.ndarray | torch.Tensor | None = None,
    m_per_unit: float | None = None,
) -> str:
    """Build a self-contained HTML page with an interactive splat / point cloud viewer.

    Args:
        params: Gaussian parameters.
        cameras: Optional training cameras, used to pick the initial view.
        max_splats: Maximum number of Gaussians embedded (bounds page size:
            about 58 bytes per Gaussian after base64, with classes).
        background: CSS background color.
        points: Optional dense point cloud ``(xyz, rgb)`` or ``(xyz, rgb, index)``
            in scene coordinates (see
            :func:`geosplat.pointcloud.dense_points_from_splat` with
            ``return_index=True``); adds the **Point cloud** view. With
            ``index``, each point takes the class of its Gaussian.
        max_points: Maximum number of points embedded (about 35 bytes per point
            after base64, with classes).
        classes: Add the **Classes** color mode (solid / ground vegetation / trees).
        region: Per-Gaussian mask of the reconstructed area (see
            :func:`geosplat.pointcloud.processed_region`); Gaussians outside keep
            their colors in the Classes mode. ``None`` classifies all.
        m_per_unit: Meters per scene unit (``geo.scale``) for metric object
            segmentation sizes; ``None`` uses sizes relative to the Gaussian spacing.

    Returns:
        HTML document as a string.
    """
    splat, order = splat_bytes(params, max_splats, return_order=True)  # type: ignore[misc]
    pos, look = _initial_view(cameras)
    js: dict[str, object] = {"splat": base64.b64encode(splat).decode("ascii"), "ptsPos": None, "ptsCol": None,
                             "spacing": 0, "clsS": None, "clsP": None}
    n = len(params["means"])
    region_np = np.ones(n, bool) if region is None else np.asarray(
        region.cpu() if isinstance(region, torch.Tensor) else region, bool)
    scene_classes = None
    if classes:
        from .classify import classify_scene

        scene_classes = classify_scene(params, region_np, m_per_unit)
        js.update(tg=round(float(scene_classes["ground_greenness"]), 4),  # type: ignore[arg-type]
                  mv=round(float(scene_classes["min_evidence"]), 2),  # type: ignore[arg-type]
                  objVotes=[round(float(v), 3) for v in scene_classes["object_votes"]],  # type: ignore[union-attr]
                  nobj=int(np.asarray(scene_classes["ids"]).max()),
                  clsS=base64.b64encode(class_bytes(scene_classes, region_np, order)).decode("ascii"))
    if points is not None and len(points[0]):
        index = points[2] if len(points) > 2 else None
        p_bytes, c_bytes, spacing, index = points_bytes(points[0], points[1], max_points, index)
        js.update(ptsPos=base64.b64encode(p_bytes).decode("ascii"), ptsCol=base64.b64encode(c_bytes).decode("ascii"),
                  spacing=round(spacing, 6))
        if scene_classes is not None and index is not None:
            js["clsP"] = base64.b64encode(class_bytes(scene_classes, region_np, index)).decode("ascii")
    return (_PAGE.replace("__BG__", background).replace("__THREE__", THREE_URL).replace("__VIEWER__", VIEWER_URL)
            .replace("__POS__", json.dumps(pos)).replace("__LOOK__", json.dumps(look))
            .replace("__GR__", repr(GREENNESS_RANGE)).replace("__DATA__", json.dumps(js)))


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>geosplat viewer</title>
<style>html,body{margin:0;height:100%;background:__BG__;overflow:hidden;font:12px sans-serif;color:#ccc}
#hint{position:absolute;left:8px;bottom:6px;opacity:.7}
#ui{position:absolute;left:8px;top:8px;display:flex;flex-direction:column;gap:6px;z-index:10;width:380px;max-width:calc(100vw - 16px)}
.row{display:flex;gap:4px;align-items:center;flex-wrap:wrap}
#ui button{background:#2a3038;color:#ddd;border:1px solid #556;border-radius:4px;padding:4px 9px;cursor:pointer}
#ui button.on{background:#ff2fb3;color:#fff;border-color:#ff2fb3}
#ui button.i{padding:0 6px;border-radius:50%;font:italic bold 11px serif;line-height:16px}
.panel{background:rgba(20,24,30,.85);border:1px solid #445;border-radius:6px;padding:6px 8px;display:none}
.panel label{display:flex;align-items:center;gap:6px;margin:4px 0}
.panel input[type=range]{flex:1;min-width:140px} .val{width:40px;text-align:right;font-variant-numeric:tabular-nums}
.name{width:92px;white-space:nowrap} .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
.info{display:none;margin:0 0 6px;padding:4px 6px;background:rgba(255,255,255,.06);border-radius:4px;line-height:1.35}</style>
<script type="importmap">{"imports":{"three":"__THREE__","@mkkellogg/gaussian-splats-3d":"__VIEWER__"}}</script>
</head><body>
<div id="ui">
 <div class="row"><button id="vS" class="on">Splats</button><button id="vP">Point cloud</button>
  <span style="width:8px"></span><button id="cR" class="on">RGB</button><button id="cC">Classes</button></div>
 <div class="panel" id="sizeP"><label><span class="name">point size</span>
  <input id="size" type="range" min="0.1" max="5" step="0.05" value="1"><span class="val" id="sizeV"></span></label></div>
 <div class="panel" id="clsP">
  <div class="row" style="margin-bottom:4px"><button id="kT" class="on">Trees</button><button id="kO">Objects</button></div>
  <div id="treeOpts">
  <label><span class="name">structures</span>
   <input id="ts" type="range" min="0" max="1" step="0.01"><span class="val" id="tsV"></span><button class="i" data-info="iTs">i</button></label>
  <div class="info" id="iTs">Structure score an object needs to be a <b>structure</b>; with less it is a <b>tree</b>.
   The score uses everything the Gaussians tell: opacity and translucency, view-dependent color, size and shape,
   orientation of their surfaces, brightness texture, how scattered and coherent they are around each one (three
   scales) and the object's height. Trees, with or without leaves, are translucent, view-dependent and scattered.
   Move right for fewer structures.</div>
  <label><span class="name">grass</span>
   <input id="tg" type="range" min="-0.3" max="0.3" step="0.005"><span class="val" id="tgV"></span><button class="i" data-info="iTg">i</button></label>
  <div class="info" id="iTg">Greenness (2G &minus; R &minus; B) from which the ground counts as <b>ground vegetation</b>;
   below it, <b>bare ground</b>. Move left for more grass.</div>
  </div>
  <label><span class="name">opacity</span>
   <input id="op" type="range" min="0" max="1" step="0.05" value="0.6"><span class="val" id="opV"></span><button class="i" data-info="iOp">i</button></label>
  <div class="info" id="iOp">How strongly the class colors cover the photo colors (0: photo colors only).</div>
  <div id="legend" style="margin-top:4px"></div>
  <button id="reset" style="margin-top:4px">Reset</button>
 </div>
</div>
<div id="hint">left-drag: orbit &middot; right-drag: pan &middot; wheel: zoom</div>
<script type="module">
import * as THREE from "three";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
const D = __DATA__;
const GR = __GR__;
const decode = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const $ = id => document.getElementById(id);
const splatRGB = decode(D.splat);
const clsS = D.clsS ? decode(D.clsS) : null, clsP = D.clsP ? decode(D.clsP) : null;
const COLORS = [[217, 158, 102], [190, 240, 80], [10, 110, 30], [90, 120, 230]];
const NAMES = ["bare ground", "ground vegetation", "trees", "structures"];
let showPoints = false, showClasses = false, objectsMode = false;
const R = 4;                                             // class bytes per record
function objColor(id) {                                   // distinct, stable color per object id
  const h = (id * 0.618034) % 1, s = 0.75, v = 0.95, i = Math.floor(h * 6), f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  const c = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
  return c.map(x => x * 255);
}

const threeScene = new THREE.Scene();
let cloud = null, ptsRGB = null;
if (D.ptsPos) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(decode(D.ptsPos).buffer), 3));
  ptsRGB = decode(D.ptsCol);
  g.setAttribute("color", new THREE.BufferAttribute(ptsRGB.slice(), 3, true));
  cloud = new THREE.Points(g, new THREE.PointsMaterial({size: 1.5 * D.spacing, vertexColors: true, sizeAttenuation: true}));
  cloud.visible = false;
  threeScene.add(cloud);
}
const viewer = new GaussianSplats3D.Viewer({
  cameraUp: [0, -1, 0], initialCameraPosition: __POS__, initialCameraLookAt: __LOOK__,
  sharedMemoryForWorkers: false, gpuAcceleratedSort: false, dynamicScene: false, threeScene: threeScene,
});
const sceneOpts = {format: GaussianSplats3D.SceneFormat.Splat, splatAlphaRemovalThreshold: 5};
const blobUrl = bytes => URL.createObjectURL(new Blob([bytes], {type: "application/octet-stream"}));

// Trees: 0 bare ground, 1 ground vegetation, 2 trees, 3 structures. An object standing on the ground is a
// structure when enough cues say so (D.objVotes), else a tree; loose standing Gaussians count as trees when
// they look like foliage, else as ground. Objects: one color per object, the rest keeps RGB.
// Records outside the reconstructed area keep their RGB colors.
function recolor(rgb, stride, off, C, n, out) {
  const tg = +$("tg").value, tf = +$("ts").value, a = +$("op").value, counts = [0, 0, 0, 0];
  let inObj = 0, inRegion = 0;
  for (let i = 0; i < n; i++) {
    const j = i * stride + off, o = i * R, flags = C[o + 1];
    out[j] = rgb[j]; out[j + 1] = rgb[j + 1]; out[j + 2] = rgb[j + 2];
    if (!(flags & 1)) continue;
    inRegion++;
    const id = C[o + 2] | (C[o + 3] << 8);
    let col;
    if (objectsMode) {
      if (!id) continue;
      inObj++; col = objColor(id);
    } else {
      let c;
      if (id) c = D.objVotes[id] >= tf ? 3 : 2;
      else if ((flags & 2) && (flags & 4)) c = 2;
      else c = C[o] / 255 * 2 * GR - GR >= tg ? 1 : 0;
      counts[c]++; col = COLORS[c];
    }
    for (let k = 0; k < 3; k++) out[j + k] = (1 - a) * rgb[j + k] + a * col[k];
  }
  if (objectsMode) {
    $("legend").innerHTML = `${D.nobj} objects (${(100 * inObj / (inRegion || 1)).toFixed(0)}% of the area); ground in RGB`;
  } else {
    const total = counts.reduce((x, y) => x + y, 0) || 1;
    $("legend").innerHTML = counts.map((v, i) => `<div><span class="sw" style="background:rgb(${COLORS[i]})"></span>${NAMES[i]} ${(100 * v / total).toFixed(0)}%</div>`).join("");
  }
}

let busy = false, again = false, splatsClassed = false;
async function reloadSplats() {
  if (busy) { again = true; return; }
  busy = true;
  let bytes = splatRGB;
  if (showClasses && clsS) { bytes = splatRGB.slice(); recolor(splatRGB, 32, 24, clsS, splatRGB.length / 32, bytes); }
  try {
    await viewer.removeSplatScene(0, false);
    await viewer.addSplatScene(blobUrl(bytes), {...sceneOpts, showLoadingUI: false});
    splatsClassed = showClasses;
    applyVisibility();
  } finally {
    busy = false;
    if (again) { again = false; reloadSplats(); }
  }
}
function updatePoints() {
  if (!cloud) return;
  const attr = cloud.geometry.getAttribute("color");
  if (showClasses && clsP) recolor(ptsRGB, 3, 0, clsP, ptsRGB.length / 3, attr.array);
  else attr.array.set(ptsRGB);
  attr.needsUpdate = true;
}
function applyVisibility() {
  const mesh = viewer.getSplatMesh();
  if (mesh) mesh.visible = !showPoints;
  if (cloud) cloud.visible = showPoints;
}
function labels() {
  for (const id of ["tg", "op", "size"]) $(id + "V").textContent = (+$(id).value).toFixed(id === "tg" ? 3 : 2);
  $("tsV").textContent = Math.round(100 * $("ts").value) + "%";
}
function refresh(rebuildSplats) {
  $("vP").classList.toggle("on", showPoints); $("vS").classList.toggle("on", !showPoints);
  $("cC").classList.toggle("on", showClasses); $("cR").classList.toggle("on", !showClasses);
  $("sizeP").style.display = showPoints ? "block" : "none";
  $("clsP").style.display = showClasses && (showPoints ? clsP : clsS) ? "block" : "none";
  $("kO").classList.toggle("on", objectsMode); $("kT").classList.toggle("on", !objectsMode);
  $("treeOpts").style.display = objectsMode ? "none" : "block";
  labels();
  applyVisibility();
  if (showPoints) updatePoints();
  else if (rebuildSplats || splatsClassed !== showClasses) reloadSplats();
}
function setDefaults() { $("tg").value = D.tg; $("ts").value = D.mv; $("op").value = 0.6; }

viewer.addSplatScene(blobUrl(splatRGB), {...sceneOpts, showLoadingUI: true})
  .then(() => {
    viewer.start();
    if (!cloud) $("vP").style.display = "none";
    if (!clsS) { $("cR").style.display = "none"; $("cC").style.display = "none"; }
    setDefaults();
    $("vS").onclick = () => { showPoints = false; refresh(false); };
    $("vP").onclick = () => { showPoints = true; refresh(false); };
    $("cR").onclick = () => { showClasses = false; refresh(false); };
    $("cC").onclick = () => { showClasses = true; refresh(false); };
    $("kT").onclick = () => { objectsMode = false; refresh(true); };
    $("kO").onclick = () => { objectsMode = true; refresh(true); };
    $("size").oninput = e => { cloud.material.size = 1.5 * D.spacing * e.target.value; labels(); };
    for (const id of ["tg", "ts", "op"]) {
      $(id).oninput = () => refresh(false);      // points: live; splats: labels only
      $(id).onchange = () => refresh(true);      // splats rebuilt on release
    }
    $("reset").onclick = () => { setDefaults(); refresh(true); };
    for (const b of document.querySelectorAll("button.i")) {
      b.onclick = e => { e.preventDefault(); const el = $(b.dataset.info); el.style.display = el.style.display === "block" ? "none" : "block"; };
    }
    const h = location.hash;
    showPoints = h.includes("points"); showClasses = h.includes("classes") || h.includes("objects");
    objectsMode = h.includes("objects");
    refresh(false);
  })
  .catch(e => { document.body.insertAdjacentHTML("beforeend", "<pre style='color:#f88'>" + e + "</pre>"); });
</script></body></html>"""


def save_viewer(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    path: str | Path,
    cameras: list[Camera] | None = None,
    max_splats: int | None = 400_000,
    points: tuple[np.ndarray, ...] | None = None,
    max_points: int | None = 400_000,
    classes: bool = True,
    region: np.ndarray | torch.Tensor | None = None,
    m_per_unit: float | None = None,
) -> Path:
    """Write a standalone HTML viewer (open it in any modern browser).

    Args:
        params: Gaussian parameters.
        path: Output ``.html`` path.
        cameras: Optional training cameras for the initial view.
        max_splats: Maximum number of Gaussians embedded.
        points: Optional dense point cloud ``(xyz, rgb[, index])`` in scene
            coordinates for the **Point cloud** view.
        max_points: Maximum number of points embedded.
        classes: Add the **Classes** color mode.
        region: Per-Gaussian mask of the reconstructed area (classified part).
        m_per_unit: Meters per scene unit (``geo.scale``) for object segmentation.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(viewer_html(params, cameras, max_splats, points=points, max_points=max_points, classes=classes,
                                region=region, m_per_unit=m_per_unit))
    return path


def show(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor] | str | Path,
    cameras: list[Camera] | None = None,
    max_splats: int | None = 300_000,
    height: int = 600,
    points: tuple[np.ndarray, ...] | None = None,
    max_points: int | None = 300_000,
    classes: bool = True,
    region: np.ndarray | torch.Tensor | None = None,
    m_per_unit: float | None = None,
) -> object:
    """Display an interactive splat viewer in a Jupyter / Colab output cell.

    Args:
        params: Gaussian parameters, or a path to a ``splat.ply``.
        cameras: Optional training cameras for the initial view.
        max_splats: Maximum number of Gaussians sent to the browser.
        height: Viewer height in pixels.
        points: Optional dense point cloud ``(xyz, rgb[, index])`` in scene
            coordinates for the **Point cloud** view.
        max_points: Maximum number of points sent to the browser.
        classes: Add the **Classes** color mode.
        region: Per-Gaussian mask of the reconstructed area (classified part).
        m_per_unit: Meters per scene unit (``geo.scale``) for object segmentation.

    Returns:
        An ``IPython.display.HTML`` object (displayed when it is the last
        expression of a cell).
    """
    from IPython.display import HTML

    if isinstance(params, (str, Path)):
        from .gaussians import load_ply

        params = load_ply(params)
    page = html.escape(viewer_html(params, cameras, max_splats, points=points, max_points=max_points,
                                   classes=classes, region=region, m_per_unit=m_per_unit), quote=True)
    return HTML(
        f'<iframe srcdoc="{page}" style="width:100%;height:{height}px;border:0;border-radius:6px" '
        f'allow="fullscreen"></iframe>'
    )
