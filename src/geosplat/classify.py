"""Per-Gaussian features to tell solid surfaces (walls, ground, rock) from vegetation.

Computed once and thresholded interactively in the viewer. Color separates
vegetation from built surfaces; the Gaussians' own shape and appearance
separate trees from ground vegetation:

* **greenness**: excess-green index ``2G - R - B`` of the Gaussian color
  (vegetation is green; most built surfaces are not).
* **opacity**: canopies are fitted by semi-transparent Gaussians, solid
  surfaces by opaque ones.
* **flatness**: ``1 - s3 / s2`` with ``s1 >= s2 >= s3`` the Gaussian scales.
  Solid surfaces are fitted by flat, disc-like Gaussians; foliage by blobs.
* **elongation**: ``1 - s2 / s1``. Branches and twigs give needle-like Gaussians.
* **view dependence**: energy of the higher spherical-harmonic coefficients
  relative to the base color, as ``v / (1 + v)``. Leaves change appearance
  with the viewpoint; matte ground and walls barely do.
* **normal coherence**: how well the normals (shortest axis) of the Gaussians
  in a neighbourhood agree, ``(l1 - l2) / l1`` from the eigenvalues of their
  opacity-weighted normal tensor. Close to 1 on walls and ground, low in
  canopies where Gaussians point every way. Computed for three neighbourhood
  sizes (``small``, ``medium``, ``large``).
* **scatter**: ``l3 / l1`` from the eigenvalues of the covariance of the
  Gaussian centers in a neighbourhood. Close to 0 on walls and ground (centers
  on a thin layer), high in canopies (centers spread in 3D). Same three sizes.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from .sh import sh_to_rgb

NEIGHBOURHOODS = (3.0, 6.0, 12.0)   # voxel sizes, in typical Gaussian spacings
# Features that separate trees from ground vegetation, and the side of the threshold where trees are.
TREE_FEATURES = {"opacity": "<=", "flatness": "<=", "elongation": ">=", "view_dependence": ">=",
                 "scatter": ">=", "coherence": "<="}
NEIGHBOURHOOD_FEATURES = ("scatter", "coherence")


def _quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrices from unit quaternions.

    Args:
        q: Quaternions ``[N, 4]`` (``wxyz``).

    Returns:
        Rotations ``[N, 3, 3]``.
    """
    w, x, y, z = torch.nn.functional.normalize(q, dim=1).unbind(1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], 1).view(-1, 3, 3)


def typical_spacing(means: torch.Tensor) -> float:
    """Typical distance between neighbouring Gaussian centers (surface-like scenes).

    Args:
        means: Centers ``[N, 3]``.

    Returns:
        Spacing in scene units.
    """
    gen = torch.Generator(device="cpu").manual_seed(0)                 # deterministic
    sample = means[torch.randperm(len(means), generator=gen)[:200_000].to(means.device)].float()
    lo, hi = torch.quantile(sample, 0.05, dim=0), torch.quantile(sample, 0.95, dim=0)
    span = torch.sort(hi - lo).values[1:]                       # two largest extents
    return float(torch.sqrt(span.prod() / len(means)).clamp_min(1e-6))


def _neighbourhood(
    means: torch.Tensor, normals: torch.Tensor, weight: torch.Tensor, voxel: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normal coherence and center scatter of each Gaussian's voxel.

    Args:
        means: Centers ``[N, 3]``.
        normals: Unit normals ``[N, 3]``.
        weight: Per-Gaussian weights ``[N]`` (opacity).
        voxel: Voxel size.

    Returns:
        Tuple ``(coherence [N], scatter [N], known [N])``, coherence and scatter
        in ``[0, 1]``; ``known`` is False for voxels with fewer than 4 Gaussians.
    """
    key = torch.floor((means - means.min(0).values) / voxel).long()
    _, inv = torch.unique(key, dim=0, return_inverse=True)
    n_vox = int(inv.max()) + 1
    dev = means.device
    count = torch.zeros(n_vox, device=dev).index_add_(0, inv, torch.ones_like(weight))
    wsum = torch.zeros(n_vox, device=dev).index_add_(0, inv, weight).clamp_min(1e-12)
    outer = (normals[:, :, None] * normals[:, None, :]).reshape(-1, 9) * weight[:, None]
    ntensor = torch.zeros(n_vox, 9, device=dev).index_add_(0, inv, outer).view(-1, 3, 3)
    ev = torch.linalg.eigvalsh(ntensor)                          # ascending
    coh = (ev[:, 2] - ev[:, 1]) / ev[:, 2].clamp_min(1e-12)
    mean = torch.zeros(n_vox, 3, device=dev).index_add_(0, inv, means * weight[:, None]) / wsum[:, None]
    d = means - mean[inv]
    cov = torch.zeros(n_vox, 9, device=dev).index_add_(
        0, inv, (d[:, :, None] * d[:, None, :]).reshape(-1, 9) * weight[:, None]).view(-1, 3, 3)
    ec = torch.linalg.eigvalsh(cov / wsum[:, None, None]).clamp_min(0)
    scatter = ec[:, 0] / ec[:, 2].clamp_min(1e-12)
    known = count >= 4
    return coh[inv], scatter[inv], known[inv]


@torch.no_grad()
def gaussian_features(params: torch.nn.ParameterDict | dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Surface / vegetation features of every Gaussian.

    Args:
        params: Gaussian parameters (unconstrained form, as trained).

    Returns:
        ``{"greenness", "opacity", "flatness", "elongation", "view_dependence",
        "axis_aligned", "luminance", "size", "coherence_<size>", "scatter_<size>"}``
        with ``<size>`` in ``small``, ``medium``, ``large``; each ``[N]``
        float32. ``axis_aligned`` is 1 when the Gaussian's normal is within 20
        degrees of vertical or horizontal (walls, roofs, ground), ``size`` the
        log of its largest scale over the typical spacing; greenness is in
        ``[-2, 2]``, the others in ``[0, 1]``.
    """
    means = params["means"].detach().float()
    scales = torch.exp(params["scales"].detach().float())
    opacity = torch.sigmoid(params["opacities"].detach().float()).reshape(-1)
    rgb = sh_to_rgb(params["sh0"].detach().float()[:, 0]).clamp(0, 1)
    greenness = 2 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]
    s = torch.sort(scales, dim=1, descending=True).values
    flatness = 1 - s[:, 2] / s[:, 1].clamp_min(1e-12)
    elongation = 1 - s[:, 1] / s[:, 0].clamp_min(1e-12)
    if "shN" in params and params["shN"].numel():
        v = params["shN"].detach().float().flatten(1).norm(dim=1) / params["sh0"].detach().float().flatten(1).norm(
            dim=1).clamp_min(1e-6)
        view_dependence = v / (1 + v)
    else:
        view_dependence = torch.zeros_like(opacity)
    R = _quat_to_rot(params["quats"].detach().float())
    normals = torch.gather(R, 2, scales.argmin(1)[:, None, None].expand(-1, 3, 1))[..., 0]
    spacing = typical_spacing(means)
    up_dot = normals[:, 1].abs()                                     # scene up = -Y
    axis_aligned = ((up_dot > 0.94) | (up_dot < 0.34)).float()         # within 20 deg of vertical or horizontal
    luminance = rgb.mean(1)
    size = torch.log(s[:, 0] / max(typical_spacing(means), 1e-12))
    out = {"greenness": greenness, "opacity": opacity, "flatness": flatness, "elongation": elongation,
           "axis_aligned": axis_aligned, "luminance": luminance, "size": size,
           "view_dependence": view_dependence}
    # Largest first: a sparse smaller neighbourhood takes the value of the next larger one.
    prev = None
    for name, k in reversed(list(zip(("small", "medium", "large"), NEIGHBOURHOODS))):
        coh, sc, known = _neighbourhood(means, normals, opacity, k * spacing)
        if prev is not None:
            coh, sc = torch.where(known, coh, prev[0]), torch.where(known, sc, prev[1])
        else:                                                    # largest still sparse: neutral values
            coh, sc = torch.where(known, coh, torch.zeros_like(coh)), torch.where(known, sc, torch.full_like(sc, 0.5))
        out[f"coherence_{name}"], out[f"scatter_{name}"] = coh, sc
        prev = (coh, sc)
    return {k: v.cpu().numpy().astype(np.float32) for k, v in out.items()}



def otsu(values: np.ndarray, weights: np.ndarray | None = None, bins: int = 256) -> tuple[float, float]:
    """Otsu threshold: the cut that maximizes the between-class variance of a histogram.

    Args:
        values: Samples.
        weights: Optional sample weights.
        bins: Histogram bins (between the 1st and 99th percentiles).

    Returns:
        Tuple ``(threshold, separability)``; separability is the between-class
        variance divided by the total variance (0: no split, 1: two spikes).
    """
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        return float(lo), 0.0
    hist, edges = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi), weights=weights)
    p = hist / max(hist.sum(), 1e-12)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centers)
    between = (mu[-1] * w0 - mu) ** 2 / np.clip(w0 * (1 - w0), 1e-12, None)
    total = float((p * (centers - mu[-1]) ** 2).sum())
    best = int(np.argmax(between[:-1]))
    return float(centers[best]), float(between[best] / max(total, 1e-12))


def _rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Fraction of ``reference`` below each value (empirical CDF).

    Args:
        values: Values to rank.
        reference: Reference sample.

    Returns:
        Ranks in ``[0, 1]``.
    """
    ref = np.sort(reference)
    return np.searchsorted(ref, values, side="right") / max(len(ref), 1)


def _filter2d(grid: np.ndarray, k: int, func: object) -> np.ndarray:
    """Min / max filter of a 2D grid over a ``(2k+1) x (2k+1)`` window, ignoring NaN.

    Args:
        grid: 2D array (NaN = empty).
        k: Window half-size in cells.
        func: ``np.nanmin`` or ``np.nanmax``.

    Returns:
        Filtered grid (NaN where the whole window is empty).
    """
    W, H = grid.shape
    padded = np.pad(grid, k, constant_values=np.nan)
    out = np.full_like(grid, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(W):                                  # row by row: bounded memory
            win = np.lib.stride_tricks.sliding_window_view(padded[i:i + 2 * k + 1], (2 * k + 1, 2 * k + 1))[0]
            out[i] = func(win.reshape(H, -1), axis=1)  # type: ignore[operator]
    return out


def _components(keys: np.ndarray, reach: int = 1) -> np.ndarray:
    """Connected components of occupied voxels.

    Args:
        keys: Integer voxel coordinates ``[V, 3]`` (unique rows).
        reach: Voxels up to this many steps apart along each axis are
            neighbours (1: 26-neighbourhood; 2 also bridges one-voxel gaps).

    Returns:
        Component label per voxel ``[V]`` (the smallest voxel index of its component).
    """
    n = len(keys)
    span = keys.max(0) - keys.min(0) + 2 * reach + 1
    base = keys - keys.min(0) + reach
    code = (base[:, 0] * span[1] + base[:, 1]) * span[2] + base[:, 2]
    order = np.argsort(code)
    sorted_code = code[order]
    a_all, b_all = [], []
    steps = range(-reach, reach + 1)
    for dx in steps:
        for dy in steps:
            for dz in steps:
                if (dx, dy, dz) <= (0, 0, 0):              # each neighbour pair once
                    continue
                c = code + (dx * span[1] + dy) * span[2] + dz
                pos = np.clip(np.searchsorted(sorted_code, c), 0, n - 1)
                hit = sorted_code[pos] == c
                a_all.append(np.flatnonzero(hit))
                b_all.append(order[pos[hit]])
    a, b = np.concatenate(a_all), np.concatenate(b_all)
    labels = np.arange(n)
    while True:                                            # hooking + pointer jumping (union-find)
        la, lb = labels[a], labels[b]
        lo, hi = np.minimum(la, lb), np.maximum(la, lb)
        diff = lo != hi
        if not diff.any():
            break
        np.minimum.at(labels, hi[diff], lo[diff])
        while True:
            nxt = labels[labels]
            if np.array_equal(nxt, labels):
                break
            labels = nxt
    return labels


def _voxel_lookup(keys: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Index of each query voxel in ``keys``, or -1.

    Args:
        keys: Integer voxel coordinates ``[V, 3]`` (unique rows).
        queries: Integer voxel coordinates ``[M, 3]``.

    Returns:
        ``[M]`` indices into ``keys`` (-1 where absent).
    """
    lo = np.minimum(keys.min(0), queries.min(0)) - 1
    span = np.maximum(keys.max(0), queries.max(0)) - lo + 2
    code = lambda v: ((v[:, 0] - lo[0]) * span[1] + (v[:, 1] - lo[1])) * span[2] + (v[:, 2] - lo[2])  # noqa: E731
    kc = code(keys)
    order = np.argsort(kc)
    q = code(queries)
    pos = np.clip(np.searchsorted(kc[order], q), 0, len(kc) - 1)
    hit = kc[order][pos] == q
    return np.where(hit, order[pos], -1)


@torch.no_grad()
def segment_objects(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    region: np.ndarray | None = None,
    m_per_unit: float | None = None,
    min_gaussians: int = 50,
    ground_quantile: float = 0.1,
    noise_factor: float = 3.0,
    voxel_m: float = 0.5,
    reach: int = 1,
    attach: int = 2,
    ground_window_m: float = 15.0,
    size_voxel: bool = True,
    min_voxel_gaussians: int = 3,
) -> dict[str, np.ndarray | float]:
    """Terrain, height above it, and separate objects (trees, buildings, walls...) of a splat.

    * **Terrain**: each 3 m map cell gets a low percentile (``ground_quantile``)
      of the heights of its opaque Gaussians; the terrain is the morphological
      opening of that grid over ``ground_window_m`` (trees and walls narrower
      than the window drop out, slopes stay).
    * **Standing**: Gaussians higher above the terrain than ``max(0.5 m,
      noise_factor x median ground roughness)``.
    * **Objects**: standing Gaussians are put in voxels whose size follows the
      Gaussians (at least ``voxel_m``, 1.5 typical spacings and the median
      Gaussian size), so sparse and dense splats behave alike. A voxel is solid
      when it holds ``min_voxel_gaussians`` Gaussians whose opacities add up
      to the Otsu threshold of the voxel masses (in log): the thin, semi-transparent fuzz of the splat does
      not glue objects to the ground, dense canopies and walls do count.
      Solid voxels up to ``reach`` steps apart form one object, and every
      standing Gaussian joins the nearest object within ``attach`` voxels.

    Scene coordinates are upright with up = -Y. Without ``m_per_unit`` sizes
    are relative to the typical Gaussian spacing.

    Args:
        params: Gaussian parameters (scene coordinates).
        region: Per-Gaussian mask of the reconstructed area (others: no object).
        m_per_unit: Meters per scene unit (``geo.scale``), if georeferenced.
        min_gaussians: Smaller clusters are not objects.
        ground_quantile: Height percentile of a cell taken as its ground.
        noise_factor: The standing threshold is at least this many times the
            median ground roughness of the scene.
        voxel_m: Smallest voxel size in meters (with ``m_per_unit``).
        reach: Solid voxels this many steps apart belong to the same object.
        attach: Standing Gaussians join the nearest object within this many voxels.
        size_voxel: Let the voxel grow with the median Gaussian size (sparse splats).
        min_voxel_gaussians: Gaussians a solid voxel needs besides its opacity mass
            (lone Gaussians of sparse splats do not make solid voxels).
        ground_window_m: Window (meters) of the morphological opening that
            estimates the terrain; objects narrower than it are not ground.

    Returns:
        ``{"ids": [N] int32 (0: no object; 1, 2, ... largest first),
        "height": [N] float32 height above the terrain in meters (scene units
        without ``m_per_unit``; NaN outside the region), "standing": [N] bool,
        "standing_height": threshold used}``.
    """
    means = params["means"].detach().float().cpu().numpy()
    opacity = torch.sigmoid(params["opacities"].detach().float()).reshape(-1).cpu().numpy()
    scale = torch.exp(params["scales"].detach().float()).max(1).values.cpu().numpy()
    n = len(means)
    region = np.ones(n, bool) if region is None else np.asarray(region, bool)
    out: dict[str, np.ndarray | float] = {"ids": np.zeros(n, np.int32), "height": np.full(n, np.nan, np.float32),
                                          "standing": np.zeros(n, bool), "standing_height": 0.0}
    idx = np.flatnonzero(region)
    if len(idx) < min_gaussians:
        return out
    to_m = m_per_unit or 1.0
    sp = typical_spacing(params["means"].detach()[torch.as_tensor(idx)])
    if m_per_unit:
        cell, h0 = 3.0 / m_per_unit, 0.5 / m_per_unit
    else:
        cell, h0 = 40 * sp, 6 * sp
    p = means[idx]
    up = -p[:, 1]
    gx = np.floor((p[:, 0] - p[:, 0].min()) / cell).astype(np.int64)
    gz = np.floor((p[:, 2] - p[:, 2].min()) / cell).astype(np.int64)
    W, H = gx.max() + 1, gz.max() + 1
    solid = opacity[idx] > 0.5
    cid, h = (gx * H + gz)[solid], up[solid]
    order = np.lexsort((h, cid))
    cid, h = cid[order], h[order]
    cells, start, count = np.unique(cid, return_index=True, return_counts=True)
    lowest = np.full(W * H, np.nan)
    lowest[cells] = h[start + (ground_quantile * (count - 1)).astype(np.int64)]
    mid = np.full(W * H, np.nan)
    mid[cells] = h[start + (0.5 * (count - 1)).astype(np.int64)]
    k = max(1, int(round(ground_window_m / 3.0 / 2))) if m_per_unit else 3
    grid = lowest.reshape(W, H)
    terrain = _filter2d(_filter2d(grid, k, np.nanmin), k, np.nanmax)
    terrain = np.where(np.isfinite(terrain), terrain, np.nanmin(up))
    spread = (mid - lowest)[np.isfinite(lowest)]
    thr = max(h0, noise_factor * float(np.median(spread)) if len(spread) else h0)
    hag = up - terrain[gx, gz]
    standing = hag > thr
    out["height"][idx] = hag * to_m          # type: ignore[index]
    out["standing"][idx] = standing           # type: ignore[index]
    out["standing_height"] = thr * to_m
    if standing.sum() < min_gaussians:
        return out
    st = idx[standing]
    voxel = max(voxel_m / to_m if m_per_unit else 6 * sp, 1.5 * sp,
                float(np.median(scale[st])) if size_voxel else 0.0)
    vox = np.floor(means[st] / voxel).astype(np.int64)
    keys, inv = np.unique(vox, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    mass = np.bincount(inv, weights=opacity[st], minlength=len(keys))
    count_v = np.bincount(inv, minlength=len(keys))
    t, _ = otsu(np.log(mass + 1e-6))
    dense = (mass >= np.exp(t)) & (count_v >= min_voxel_gaussians)
    if dense.sum() == 0:
        return out
    dkeys = keys[dense]
    comp = _components(dkeys, reach)
    # Each standing Gaussian joins the object of the nearest solid voxel within ``reach`` steps.
    lab = np.full(len(st), -1, np.int64)
    steps = sorted(((dx, dy, dz) for dx in range(-attach, attach + 1) for dy in range(-attach, attach + 1)
                    for dz in range(-attach, attach + 1)), key=lambda d: d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
    for d in steps:
        todo = lab < 0
        if not todo.any():
            break
        hit = _voxel_lookup(dkeys, vox[todo] + np.array(d))
        sub = np.flatnonzero(todo)
        lab[sub[hit >= 0]] = comp[hit[hit >= 0]]
    member = lab >= 0
    labels, counts = np.unique(lab[member], return_counts=True)
    big = counts >= min_gaussians
    rank = np.zeros(lab.max() + 2, np.int32)
    rank[labels[big][np.argsort(-counts[big])]] = np.arange(1, big.sum() + 1)
    out["ids"][st[member]] = rank[lab[member]]  # type: ignore[index]
    return out


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Standardize columns with median and inter-quartile range.

    Args:
        x: ``[M, K]`` values.

    Returns:
        Standardized values (clipped to +-5).
    """
    med = np.nanmedian(x, 0)
    iqr = np.nanpercentile(x, 75, 0) - np.nanpercentile(x, 25, 0)
    z = (x - med) / np.where(iqr > 1e-9, iqr, 1.0)
    return np.clip(np.nan_to_num(z), -5, 5)


def object_features(
    f: dict[str, np.ndarray], ids: np.ndarray, height: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Describe every object by the distribution of its Gaussians' properties.

    For each per-Gaussian feature (opacity, view dependence, flatness,
    elongation, size, axis alignment of normals, luminance, greenness, and
    scatter / normal coherence at three neighbourhood sizes): the median and
    the inter-quartile range over the object's Gaussians; plus the share of
    translucent Gaussians (opacity < 0.5) and the object's height above the
    terrain (90th percentile).

    Args:
        f: Per-Gaussian features (:func:`gaussian_features` plus ``"opacity"``).
        ids: Object id per Gaussian.
        height: Height above the terrain per Gaussian.

    Returns:
        Tuple ``(matrix [n_objects + 1, K] (row 0 unused), column names)``.
    """
    n_obj = int(ids.max())
    order = np.argsort(ids, kind="stable")
    bounds = np.searchsorted(ids[order], np.arange(n_obj + 2))
    keys = ["opacity", "view_dependence", "flatness", "elongation", "size", "axis_aligned", "luminance",
            "greenness"] + [f"{a}_{b}" for a in ("scatter", "coherence") for b in ("small", "medium", "large")]
    cols, names = [], []
    for k in keys:
        v = f[k][order]
        med, iqr = np.zeros(n_obj + 1), np.zeros(n_obj + 1)
        for i in range(1, n_obj + 1):
            seg = v[bounds[i]:bounds[i + 1]]
            if len(seg):
                q1, q2, q3 = np.percentile(seg, [25, 50, 75])
                med[i], iqr[i] = q2, q3 - q1
        cols += [med, iqr]
        names += [f"{k}_median", f"{k}_iqr"]
    op = f["opacity"][order]
    hg = np.nan_to_num(height[order])
    transl, top = np.zeros(n_obj + 1), np.zeros(n_obj + 1)
    for i in range(1, n_obj + 1):
        a, b = bounds[i], bounds[i + 1]
        if b > a:
            transl[i] = (op[a:b] < 0.5).mean()
            top[i] = np.percentile(hg[a:b], 90)
    cols += [transl, top]
    names += ["translucent_share", "height"]
    return np.stack(cols, 1), names


def _logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """Weighted L2-regularized logistic regression (Newton iterations).

    Args:
        x: ``[M, K]`` standardized features.
        y: ``[M]`` labels in {0, 1}.
        w: ``[M]`` sample weights.
        l2: Ridge strength (not applied to the intercept).
        iters: Newton iterations.

    Returns:
        Coefficients ``[K + 1]`` (last = intercept).
    """
    X = np.hstack([x, np.ones((len(x), 1))])
    beta = np.zeros(X.shape[1])
    reg = np.full(X.shape[1], l2)
    reg[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        g = X.T @ (w * (p - y)) + reg * beta
        Hm = (X * (w * p * (1 - p))[:, None]).T @ X + np.diag(reg + 1e-9)
        step = np.linalg.solve(Hm, g)
        beta -= step
        if np.abs(step).max() < 1e-6:
            break
    return beta


# Cues that a standing object is a built structure, and the side of the split where structures are:
# opaque Gaussians whose color does not change with the viewpoint, lying on a thin layer (low scatter)
# with coherent normals. Trees, with or without leaves, are translucent, view-dependent and scattered in 3D
# with normals pointing every way. Color is not a cue (bare trees can have the color of a building), and the
# shape of single Gaussians is not either (walls are also fitted by needle-like Gaussians along edges).
STRUCTURE_CUES = {"opacity": ">=", "view_dependence": "<=", "scatter_small": "<=", "scatter_medium": "<=",
                  "coherence_small": ">="}


@torch.no_grad()
def classify_scene(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    region: np.ndarray | None = None,
    m_per_unit: float | None = None,
    min_evidence: float = 0.7,
    seed_structure: float = 0.75,
    seed_tree: float = 0.45,
    ridge: float = 1.0,
) -> dict[str, object]:
    """Everything the viewer's Classes mode needs: terrain, objects, their votes and default thresholds.

    * Objects and standing Gaussians: :func:`segment_objects`.
    * **Structure votes** of each object: for every cue in
      :data:`STRUCTURE_CUES` (opacity, view-dependent color, neighbourhood
      scatter at two sizes, normal coherence; not color, since bare trees can look like buildings) the
      object's median is compared with the Otsu split of all objects' medians
      (weighted by the square root of their size; with fewer than 3 objects,
      the split of the standing Gaussians). Each cue votes softly (0.5 at its
      threshold, 1 or 0 one inter-quartile range of the object medians beyond
      it), with the weight of how well it splits the objects (Otsu
      separability); the **structure evidence** is the voted share of the total
      weight (0-1). An object is a structure only with evidence of at least
      ``min_evidence``.
    * **Learning from all the splat information**: objects whose cue evidence
      is clear (at least ``seed_structure``: structure; at most ``seed_tree``:
      tree) train a logistic regression on every per-object feature
      (:func:`object_features`: medians and spreads of opacity, view
      dependence, Gaussian shape and size, normal orientation, brightness,
      greenness, neighbourhood scatter and coherence at three sizes,
      translucent share, height), strongly regularized. The **structure
      score** of an object is the mean of that probability and its cue
      evidence; an object is a structure from ``min_evidence``.     * **Loose standing Gaussians** (in no object) are trees when they look like
      foliage: on the tree side of the Otsu split of at least half the cues.
    * **Ground**: ground vegetation at or above the Otsu threshold of the
      ground greenness, bare ground below.

    Args:
        params: Gaussian parameters (scene coordinates).
        region: Per-Gaussian mask of the reconstructed area.
        m_per_unit: Meters per scene unit (``geo.scale``), if georeferenced.
        min_evidence: Default structure probability (0-1) an object needs.
        seed_structure: Cue evidence from which an object is a structure seed.
        seed_tree: Cue evidence up to which an object is a tree seed.
        ridge: L2 penalty of the logistic regression, per seed object.

    Returns:
        ``{"ids", "standing", "foliage", "greenness"}`` per Gaussian, plus
        ``"object_votes"`` (structure probability per object id, index 0
        unused; the cue evidence when there are too few objects to learn),
        ``"object_evidence"`` (cue evidence),
        ``"min_evidence"``, ``"ground_greenness"``, ``"cue_thresholds"`` and
        ``"cue_weights"``.
    """
    n = len(params["means"])
    region = np.ones(n, bool) if region is None else np.asarray(region, bool)
    seg = segment_objects(params, region, m_per_unit)
    ids, standing = seg["ids"], seg["standing"]
    f = gaussian_features(params)
    f["opacity"] = torch.sigmoid(params["opacities"].detach().float()).reshape(-1).cpu().numpy()
    w = f["opacity"]
    n_obj = int(ids.max())
    out: dict[str, object] = {"ids": ids, "standing": standing, "foliage": np.zeros(n, bool),
                              "greenness": f["greenness"], "object_votes": np.zeros(n_obj + 1, np.float32),
                              "min_evidence": min_evidence, "ground_greenness": 0.0, "cue_thresholds": {},
                              "cue_weights": {}}
    ground = np.flatnonzero(region & ~standing)
    if len(ground) > 100:
        out["ground_greenness"] = otsu(f["greenness"][ground], w[ground])[0]
    st = np.flatnonzero(standing)
    if len(st) < 100:
        return out
    loose_votes = np.zeros(len(st), np.int8)
    if n_obj:
        order = np.argsort(ids, kind="stable")
        bounds = np.searchsorted(ids[order], np.arange(n_obj + 2))
        sizes = np.diff(bounds)[1:].astype(np.float64)
    votes = np.zeros(n_obj + 1, np.float64)
    weight_sum = 0.0
    for key, side in STRUCTURE_CUES.items():
        sign = 1.0 if side == ">=" else -1.0
        v = f[key]
        t, sep = otsu(v[st], w[st])
        if n_obj:
            vs = v[order]
            med = np.array([np.median(vs[bounds[i]:bounds[i + 1]]) if bounds[i + 1] > bounds[i] else np.nan
                            for i in range(1, n_obj + 1)])
            ok = np.isfinite(med)
            if n_obj >= 3:
                t, sep = otsu(med[ok], np.sqrt(sizes[ok]))
            # Soft vote in [0, 1]: 0.5 at the threshold, saturating one spread (IQR of the object medians)
            # away, weighted by how well the cue splits the objects (Otsu separability).
            q1, q3 = np.percentile(med[ok], [25, 75]) if ok.sum() >= 4 else (t - 1.0, t + 1.0)
            spread = max(float(q3 - q1), 1e-6)
            soft = np.clip(0.5 + sign * (np.nan_to_num(med, nan=t) - t) / spread, 0.0, 1.0)
            votes[1:] += sep * soft
            weight_sum += sep
        out["cue_weights"][key] = round(float(sep), 3)  # type: ignore[index]
        out["cue_thresholds"][key] = round(float(t), 4)  # type: ignore[index]
        loose_votes += (sign * (v[st] - t) > 0).astype(np.int8)
    evidence = votes / max(weight_sum, 1e-12)
    out["object_evidence"] = evidence.astype(np.float32)
    out["object_votes"] = evidence.astype(np.float32)
    # Learn from all the splat information: objects the cues are sure about (seeds) train a logistic
    # regression on every per-object feature, which then scores all objects.
    if n_obj >= 6:
        X, names = object_features(f, ids, np.asarray(seg["height"]))
        Z = _robust_z(X[1:])
        ev = evidence[1:]
        struct_seed, tree_seed = ev >= seed_structure, ev <= seed_tree
        if struct_seed.sum() >= 2 and tree_seed.sum() >= 2:
            seed = struct_seed | tree_seed
            w_obj = np.sqrt(sizes)
            y = struct_seed[seed].astype(np.float64)
            w_seed = w_obj[seed].copy()
            for cls in (0.0, 1.0):                                   # balance the two classes
                w_seed[y == cls] /= max(w_seed[y == cls].sum(), 1e-12)
            beta = _logistic(Z[seed], y, w_seed * len(y), l2=float(len(y)) * ridge)
            prob = 1 / (1 + np.exp(-np.clip(np.hstack([Z, np.ones((len(Z), 1))]) @ beta, -30, 30)))
            # Final score: mean of the learned probability and the cue evidence (neither decides alone).
            out["object_votes"] = np.concatenate([[0.0], 0.5 * (prob + ev)]).astype(np.float32)
            out["object_model"] = dict(zip(names, np.round(beta[:-1], 3).tolist()))
    foliage = np.zeros(n, bool)
    foliage[st] = loose_votes <= len(STRUCTURE_CUES) // 2      # at least half the cues on the tree side
    out["foliage"] = foliage
    return out
