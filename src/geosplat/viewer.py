"""Interactive Gaussian splat viewer for notebooks (Jupyter / Colab) and standalone HTML.

The splat is converted to the compact ``.splat`` format (32 bytes per Gaussian,
most important Gaussians first) and embedded in a small HTML page that runs the
open-source WebGL viewer `@mkkellogg/gaussian-splats-3d
<https://github.com/mkkellogg/GaussianSplats3D>`_ (three.js), loaded from a CDN.

Controls: left-drag orbit, right-drag pan, wheel zoom.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import numpy as np
import torch

from .scene import Camera
from .sh import sh_to_rgb

THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js"
VIEWER_URL = "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.6/build/gaussian-splats-3d.module.js"


@torch.no_grad()
def splat_bytes(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor], max_splats: int | None = 400_000
) -> bytes:
    """Encode Gaussians in the ``.splat`` format used by web viewers.

    Per Gaussian: position (3 x float32), scale (3 x float32), color RGBA
    (4 x uint8, alpha = opacity) and rotation (4 x uint8, ``wxyz`` mapped from
    ``[-1, 1]`` to ``[0, 255]``). Gaussians are sorted by ``opacity * volume``
    so truncation keeps the most visible ones.

    Args:
        params: Gaussian parameters (unconstrained form, as trained).
        max_splats: Keep at most this many Gaussians (``None`` keeps all).

    Returns:
        The binary ``.splat`` payload.
    """
    means = params["means"].detach().float()
    scales = torch.exp(params["scales"].detach().float())
    opacity = torch.sigmoid(params["opacities"].detach().float())
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
    return buf.tobytes()


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
) -> str:
    """Build a self-contained HTML page with an interactive splat viewer.

    Args:
        params: Gaussian parameters.
        cameras: Optional training cameras, used to pick the initial view.
        max_splats: Maximum number of Gaussians embedded (bounds page size:
            about 43 bytes per Gaussian after base64).
        background: CSS background color.

    Returns:
        HTML document as a string.
    """
    data = base64.b64encode(splat_bytes(params, max_splats)).decode("ascii")
    pos, look = _initial_view(cameras)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>geosplat viewer</title>
<style>html,body{{margin:0;height:100%;background:{background};overflow:hidden;font:12px sans-serif;color:#ccc}}
#hint{{position:absolute;left:8px;bottom:6px;opacity:.7}}</style>
<script type="importmap">{{"imports":{{"three":"{THREE_URL}","@mkkellogg/gaussian-splats-3d":"{VIEWER_URL}"}}}}</script>
</head><body><div id="hint">left-drag: orbit &middot; right-drag: pan &middot; wheel: zoom</div>
<script type="module">
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
const b64 = "{data}";
const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
const url = URL.createObjectURL(new Blob([bytes], {{type: "application/octet-stream"}}));
const viewer = new GaussianSplats3D.Viewer({{
  cameraUp: [0, -1, 0],
  initialCameraPosition: {pos},
  initialCameraLookAt: {look},
  sharedMemoryForWorkers: false,
  gpuAcceleratedSort: false,
  dynamicScene: false,
}});
viewer.addSplatScene(url, {{format: GaussianSplats3D.SceneFormat.Splat, showLoadingUI: true,
                           splatAlphaRemovalThreshold: 5}})
  .then(() => viewer.start())
  .catch(e => {{ document.body.insertAdjacentHTML("beforeend", "<pre style='color:#f88'>" + e + "</pre>"); }});
</script></body></html>"""


def save_viewer(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    path: str | Path,
    cameras: list[Camera] | None = None,
    max_splats: int | None = 400_000,
) -> Path:
    """Write a standalone HTML viewer (open it in any modern browser).

    Args:
        params: Gaussian parameters.
        path: Output ``.html`` path.
        cameras: Optional training cameras for the initial view.
        max_splats: Maximum number of Gaussians embedded.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(viewer_html(params, cameras, max_splats))
    return path


def show(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor] | str | Path,
    cameras: list[Camera] | None = None,
    max_splats: int | None = 300_000,
    height: int = 600,
) -> object:
    """Display an interactive splat viewer in a Jupyter / Colab output cell.

    Args:
        params: Gaussian parameters, or a path to a ``splat.ply``.
        cameras: Optional training cameras for the initial view.
        max_splats: Maximum number of Gaussians sent to the browser.
        height: Viewer height in pixels.

    Returns:
        An ``IPython.display.HTML`` object (displayed when it is the last
        expression of a cell).
    """
    from IPython.display import HTML

    if isinstance(params, (str, Path)):
        from .gaussians import load_ply

        params = load_ply(params)
    page = html.escape(viewer_html(params, cameras, max_splats), quote=True)
    return HTML(
        f'<iframe srcdoc="{page}" style="width:100%;height:{height}px;border:0;border-radius:6px" '
        f'allow="fullscreen"></iframe>'
    )
