"""Gaussian Splatting optimization loop."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.ops import remove
from PIL import Image
from tqdm.auto import tqdm

from .gaussians import colors_from_images, fibonacci_sphere, init_gaussians, random_point_cloud, save_ply
from .render import render
from .scene import Camera, ImageDataset, Scene, split_train_test
from .utils import get_device, logger, resolve_backend, set_seed


@dataclass
class TrainConfig:
    """Options for Gaussian Splatting training.

    Defaults follow the original 3DGS paper / gsplat ``simple_trainer``.

    Attributes:
        iterations: Number of optimization steps (7k gives a quick preview,
            30k is the paper's full schedule).
        max_resolution: Longest image side used for training, in pixels.
        sh_degree: Maximum spherical-harmonics degree for view-dependent color.
        sh_degree_interval: Steps between SH degree increments.
        ssim_lambda: Weight of the D-SSIM term in the loss.
        init_opacity: Initial Gaussian opacity.
        init_scale: Multiplier on the initial nearest-neighbour scale.
        random_init_points: Number of random points used if SfM gave none.
        background_points: Number of Gaussians initialized on a far sphere
            around the scene. They capture sky and distant background so it is
            not learned as floaters close to the cameras, which makes the model
            look much cleaner in free-viewpoint viewers. ``0`` disables it.
        background_radius: Radius of the background sphere, in multiples of
            the scene scale (camera rig radius).
        lr_means: Initial learning rate of positions (multiplied by scene scale).
        lr_means_final_factor: Final means LR as a fraction of the initial LR.
        lr_scales: Learning rate of log-scales.
        lr_quats: Learning rate of rotations.
        lr_opacities: Learning rate of logit opacities.
        lr_sh0: Learning rate of DC color.
        lr_shN: Learning rate of higher-order SH coefficients.
        refine_start_iter: Step at which densification starts.
        refine_stop_iter: Step at which densification stops (capped at half the
            iterations for short runs).
        refine_every: Densify / prune every this many steps.
        reset_every: Reset opacities every this many steps.
        grow_grad2d: 2D gradient threshold for cloning / splitting.
        prune_opa: Opacity threshold below which Gaussians are pruned.
        test_every: Hold out every ``test_every``-th image for evaluation
            (``0`` uses all images for training).
        random_background: Train with random background colors (reduces floaters
            for scenes with transparent backgrounds).
        max_gaussians: Cap on the number of Gaussians. When set, gsplat's MCMC
            densification strategy is used, which keeps memory and per-step time
            bounded (good for speed and for small GPUs). ``None`` uses the
            original 3DGS adaptive densification (unbounded).
        gaussian_budget: Safety cap for the default strategy: whenever the
            number of Gaussians exceeds it, the least opaque ones are pruned.
            ``"auto"`` derives it from free GPU memory at start-up (about
            ``bytes_per_gaussian`` each); ``None`` disables the cap.
        bytes_per_gaussian: GPU memory assumed per Gaussian (parameters,
            gradients, Adam state and rasterization buffers) for the auto budget.
        mcmc_opacity_reg: Opacity regularization weight used with ``max_gaussians``.
        mcmc_scale_reg: Scale regularization weight used with ``max_gaussians``.
        flatten_weight: Weight of a loss on the smallest scale of each Gaussian
            (relative to the scene radius) that makes Gaussians flat discs lying
            on the surfaces: thinner walls, a sharper surface for the
            orthophoto and point cloud, at a small cost in rendering quality.
            ``0`` disables it.
        selective_adam: Use gsplat's fused CUDA SelectiveAdam, which only updates
            Gaussians visible in the current view (gsplat backend only).
        progressive_resolution: Coarse-to-fine training: the first steps use
            downscaled images (1/4, then 1/2 resolution), which are much cheaper.
        progressive_steps: Fractions of ``iterations`` at which resolution
            switches from 1/4 to 1/2 and from 1/2 to full.
        early_stop: Stop when held-out PSNR stops improving (checked every
            ``eval_every`` steps after densification ends). Needs a test split.
        eval_every: Steps between early-stopping evaluations.
        early_stop_patience: Number of evaluations without improvement before stopping.
        early_stop_delta: Minimum PSNR gain (dB) counted as an improvement.
        backend: ``"auto"`` (gsplat on CUDA, torch otherwise), ``"gsplat"`` or ``"torch"``.
        device: Torch device; ``None`` selects CUDA when available.
        cache_images_on_device: Keep training images on the GPU.
        seed: Random seed.
        log_every: Steps between progress-bar updates.
        save_every: Steps between intermediate PLY snapshots (``0`` disables).
    """

    iterations: int = 7000
    max_resolution: int = 1600
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    ssim_lambda: float = 0.2
    init_opacity: float = 0.1
    init_scale: float = 1.0
    random_init_points: int = 100_000
    background_points: int = 50_000
    background_radius: float = 2.0
    lr_means: float = 1.6e-4
    lr_means_final_factor: float = 0.01
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000
    grow_grad2d: float = 2e-4
    prune_opa: float = 0.005
    test_every: int = 8
    random_background: bool = False
    max_gaussians: int | None = None
    gaussian_budget: int | str | None = "auto"
    bytes_per_gaussian: int = 4000
    mcmc_opacity_reg: float = 0.01
    mcmc_scale_reg: float = 0.01
    flatten_weight: float = 0.0
    selective_adam: bool = True
    progressive_resolution: bool = True
    progressive_steps: tuple[float, float] = (0.1, 0.25)
    early_stop: bool = False
    eval_every: int = 1000
    early_stop_patience: int = 2
    early_stop_delta: float = 0.05
    backend: Literal["auto", "gsplat", "torch"] = "auto"
    device: str | None = None
    cache_images_on_device: bool = True
    seed: int = 42
    log_every: int = 50
    save_every: int = 0


@dataclass
class TrainResult:
    """Outputs of a training run.

    Attributes:
        params: Trained Gaussian parameters.
        ply_path: Path of the exported ``.ply``.
        metrics: Evaluation metrics (PSNR, SSIM, number of Gaussians, time).
        sh_degree: SH degree of the trained model.
        backend: Rasterization backend that was used.
        train_cameras: Cameras used for training (at training resolution).
        test_cameras: Held-out cameras (at training resolution).
    """

    params: torch.nn.ParameterDict
    ply_path: Path
    metrics: dict[str, float | str]
    sh_degree: int
    backend: str
    train_cameras: list[Camera] = field(default_factory=list)
    test_cameras: list[Camera] = field(default_factory=list)


def _gaussian_1d(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalized 1D Gaussian kernel.

    Args:
        size: Kernel size in pixels.
        sigma: Standard deviation in pixels.
        device: Kernel device.
        dtype: Kernel dtype.

    Returns:
        Kernel of shape ``[size]``.
    """
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Structural similarity between two image batches (fast GPU version).

    The Gaussian window is applied as two separable 1D convolutions, and the
    five local statistics (``x, y, x², y², xy``) are filtered in one batched
    call, which is several times faster than five dense 11x11 convolutions.

    Args:
        img1: Images ``[B, C, H, W]`` in ``[0, 1]``.
        img2: Images ``[B, C, H, W]`` in ``[0, 1]``.
        window_size: Gaussian window size.

    Returns:
        Scalar mean SSIM.
    """
    B, C, H, W = img1.shape
    g = _gaussian_1d(window_size, 1.5, img1.device, img1.dtype)
    stats = torch.cat([img1, img2, img1 * img1, img2 * img2, img1 * img2], dim=1)  # [B, 5C, H, W]
    k = 5 * C
    pad = window_size // 2
    stats = F.conv2d(stats, g.view(1, 1, 1, -1).expand(k, 1, 1, window_size), padding=(0, pad), groups=k)
    stats = F.conv2d(stats, g.view(1, 1, -1, 1).expand(k, 1, window_size, 1), padding=(pad, 0), groups=k)
    mu1, mu2, e11, e22, e12 = stats.split(C, dim=1)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1, s2, s12 = e11 - mu1_sq, e22 - mu2_sq, e12 - mu12
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return ssim_map.mean()


def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Peak signal-to-noise ratio for images in ``[0, 1]``.

    Args:
        img1: First image tensor.
        img2: Second image tensor of the same shape.

    Returns:
        PSNR in dB.
    """
    mse = F.mse_loss(img1, img2).item()
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def camera_tensors(cams: list[Camera], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack camera matrices into tensors for rendering.

    Args:
        cams: Cameras (all of the same resolution).
        device: Target device.

    Returns:
        Tuple ``(viewmats [C, 4, 4], Ks [C, 3, 3])``.
    """
    viewmats = torch.tensor(np.stack([c.world_to_cam for c in cams]), dtype=torch.float32, device=device)
    Ks = torch.tensor(np.stack([c.K for c in cams]), dtype=torch.float32, device=device)
    return viewmats, Ks


def render_camera(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    camera: Camera,
    sh_degree: int,
    backend: str,
    background: torch.Tensor | None = None,
) -> torch.Tensor:
    """Render a single camera without gradients.

    Args:
        params: Gaussian parameters.
        camera: Camera to render.
        sh_degree: Active SH degree.
        backend: ``"gsplat"`` or ``"torch"``.
        background: Optional background color ``[3]``.

    Returns:
        Rendered image ``[H, W, 3]`` in ``[0, 1]``.
    """
    device = params["means"].device
    viewmats, Ks = camera_tensors([camera], device)
    with torch.no_grad():
        rgb, _, _ = render(params, viewmats, Ks, camera.width, camera.height, sh_degree, backend, background)
    return rgb[0]


def save_image(img: torch.Tensor | np.ndarray, path: str | Path) -> None:
    """Save a float ``[H, W, 3]`` image in ``[0, 1]`` as PNG/JPG.

    Args:
        img: Image tensor or array.
        path: Destination path.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


class Trainer:
    """Optimizes 3D Gaussians to reproduce the images of a :class:`Scene`.

    Args:
        scene: Scene with cameras and sparse points.
        output_dir: Folder for the PLY, renders and metrics.
        config: Training options.
    """

    def __init__(self, scene: Scene, output_dir: str | Path, config: TrainConfig | None = None) -> None:
        """Set up data, Gaussians, optimizers and the densification strategy.

        Args:
            scene: Scene with cameras and sparse points.
            output_dir: Folder for outputs.
            config: Training options (defaults to :class:`TrainConfig`).
        """
        self.cfg = config or TrainConfig()
        self.scene = scene
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_seed(self.cfg.seed)

        self.device = get_device(self.cfg.device)
        self.backend = resolve_backend(self.cfg.backend, self.device)
        logger.info("Training on %s with the %s backend", self.device, self.backend)
        if self.backend == "torch":
            logger.warning(
                "The PyTorch fallback rasterizer is slow; consider fewer iterations and a "
                "lower max_resolution (e.g. 400) when no CUDA GPU is available."
            )

        train_cams, test_cams = split_train_test(scene.cameras, self.cfg.test_every)
        cache = self.device if self.cfg.cache_images_on_device else torch.device("cpu")
        self.trainset = ImageDataset(train_cams, self.cfg.max_resolution, cache)
        self.testset = ImageDataset(test_cams, self.cfg.max_resolution, "cpu")
        logger.info("%d training / %d test images", len(self.trainset), len(self.testset))
        # Camera matrices live on the GPU once, instead of being copied every step.
        self.train_viewmats, self.train_Ks = camera_tensors(self.trainset.cameras, self.device)

        points, colors = scene.points, scene.colors
        if len(points) == 0:
            centers = np.stack([c.center for c in scene.cameras])
            points, colors = random_point_cloud(
                self.cfg.random_init_points, centers.mean(0), scene.scene_scale, self.cfg.seed
            )
        if self.cfg.background_points > 0:
            center = np.median(scene.points, axis=0) if len(scene.points) else np.zeros(3)
            radius = self.cfg.background_radius * scene.scene_scale
            bg_pts = fibonacci_sphere(self.cfg.background_points, center, radius)
            bg_cols, seen = colors_from_images(bg_pts, self.trainset.cameras, self.trainset.images)
            # Only keep background points some photo actually sees (no dome over nadir captures).
            bg_pts, bg_cols = bg_pts[seen], bg_cols[seen]
            points = np.concatenate([points, bg_pts]).astype(np.float32)
            colors = np.concatenate([colors.astype(np.uint8), bg_cols])
            logger.info("Added %d background-sphere points at radius %.2f", len(bg_pts), radius)
        self.params = init_gaussians(
            points, colors, self.cfg.sh_degree, self.cfg.init_opacity, self.cfg.init_scale, self.device
        )
        logger.info("Initialized %d Gaussians", len(self.params["means"]))

        s = scene.scene_scale
        lrs = {
            "means": self.cfg.lr_means * s,
            "scales": self.cfg.lr_scales,
            "quats": self.cfg.lr_quats,
            "opacities": self.cfg.lr_opacities,
            "sh0": self.cfg.lr_sh0,
            "shN": self.cfg.lr_shN,
        }
        self.selective = self.cfg.selective_adam and self.backend == "gsplat"
        if self.selective:
            from gsplat.optimizers import SelectiveAdam

            self.optimizers = {
                k: SelectiveAdam([{"params": self.params[k], "lr": lr, "name": k}], eps=1e-15, betas=(0.9, 0.999))
                for k, lr in lrs.items()
            }
        else:
            self.optimizers = {
                k: torch.optim.Adam([{"params": self.params[k], "lr": lr, "name": k}], eps=1e-15)
                for k, lr in lrs.items()
            }
        gamma = self.cfg.lr_means_final_factor ** (1.0 / max(self.cfg.iterations, 1))
        self.means_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizers["means"], gamma)

        self.budget = self._resolve_budget()
        stop = min(self.cfg.refine_stop_iter, self.cfg.iterations // 2)
        self.mcmc = self.cfg.max_gaussians is not None
        if self.mcmc:
            self.strategy = MCMCStrategy(
                cap_max=int(self.cfg.max_gaussians),
                refine_start_iter=self.cfg.refine_start_iter,
                refine_stop_iter=max(stop, int(0.8 * self.cfg.iterations)),
                refine_every=self.cfg.refine_every,
                verbose=False,
            )
            self.strategy.check_sanity(self.params, self.optimizers)
            self.strategy_state = self.strategy.initialize_state()
        else:
            self.strategy = DefaultStrategy(
                prune_opa=self.cfg.prune_opa,
                grow_grad2d=self.cfg.grow_grad2d,
                refine_start_iter=self.cfg.refine_start_iter,
                refine_stop_iter=stop,
                refine_every=self.cfg.refine_every,
                reset_every=self.cfg.reset_every,
                verbose=False,
            )
            self.strategy.check_sanity(self.params, self.optimizers)
            self.strategy_state = self.strategy.initialize_state(scene_scale=s)

    def _resolve_budget(self) -> int | None:
        """Compute the Gaussian count cap for the default strategy.

        Returns:
            Maximum number of Gaussians, or ``None`` for no cap.
        """
        b = self.cfg.gaussian_budget
        if b is None or self.cfg.max_gaussians is not None:
            return None
        if b != "auto":
            return int(b)
        if self.device.type != "cuda":
            return 2_000_000
        free, _ = torch.cuda.mem_get_info(self.device)
        # Images and the current model are already allocated; keep 15 % headroom.
        budget = int(free * 0.85 / self.cfg.bytes_per_gaussian)
        budget = max(budget, 2 * len(self.params["means"]), 200_000)
        logger.info("Gaussian budget: %d (%.1f GB free GPU memory)", budget, free / 1e9)
        return budget

    def _enforce_budget(self) -> None:
        """Prune the least opaque Gaussians if the count exceeds the budget."""
        n = len(self.params["means"])
        if self.budget is None or n <= self.budget:
            return
        excess = n - self.budget
        with torch.no_grad():
            idx = torch.topk(self.params["opacities"].detach(), excess, largest=False).indices
            mask = torch.zeros(n, dtype=torch.bool, device=self.device)
            mask[idx] = True
        state = {k: v for k, v in self.strategy_state.items() if isinstance(v, torch.Tensor) and len(v) == n}
        remove(self.params, self.optimizers, state, mask)
        self.strategy_state.update(state)

    def resolution_factor(self, step: int) -> int:
        """Image downscale factor at a given step (coarse-to-fine schedule).

        Args:
            step: Current training step.

        Returns:
            ``4``, ``2`` or ``1``.
        """
        if not self.cfg.progressive_resolution:
            return 1
        a, b = self.cfg.progressive_steps
        if step < a * self.cfg.iterations:
            return 4
        if step < b * self.cfg.iterations:
            return 2
        return 1

    @torch.no_grad()
    def quick_psnr(self, max_images: int = 8) -> float:
        """Mean PSNR on (up to ``max_images``) held-out images.

        Args:
            max_images: Number of test images used.

        Returns:
            Mean PSNR in dB (``nan`` without a test split).
        """
        if not len(self.testset):
            return float("nan")
        vals = []
        for i in range(min(max_images, len(self.testset))):
            cam, gt = self.testset[i]
            pred = render_camera(self.params, cam, self.cfg.sh_degree, self.backend)
            vals.append(psnr(pred, gt.to(self.device)))
        return float(np.mean(vals))

    def active_sh_degree(self, step: int) -> int:
        """SH degree in use at a given step (grows progressively).

        Args:
            step: Current training step.

        Returns:
            Active SH degree.
        """
        return min(step // self.cfg.sh_degree_interval, self.cfg.sh_degree)

    def train(self) -> TrainResult:
        """Run the optimization, then evaluate and export.

        Returns:
            A :class:`TrainResult` with the trained model and metrics.
        """
        cfg = self.cfg
        n_train = len(self.trainset)
        order = torch.randperm(n_train).tolist()
        start = time.time()
        best_psnr, bad_evals = -float("inf"), 0
        stop_after = self.strategy.refine_stop_iter
        self.stopped_at: int | None = None
        bar = tqdm(range(cfg.iterations), desc="Training", dynamic_ncols=True)
        for step in bar:
            if not order:
                order = torch.randperm(n_train).tolist()
            idx = order.pop()
            cam, gt = self.trainset[idx]
            gt = gt.to(self.device, non_blocking=True)
            viewmats, Ks = self.train_viewmats[idx : idx + 1], self.train_Ks[idx : idx + 1]
            bg = torch.rand(3, device=self.device) if cfg.random_background else None
            sh_deg = self.active_sh_degree(step)
            width, height = cam.width, cam.height
            factor = self.resolution_factor(step)
            if factor > 1:
                height, width = max(1, height // factor), max(1, width // factor)
                gt = F.interpolate(gt.permute(2, 0, 1)[None], size=(height, width), mode="area")[0].permute(1, 2, 0)
                Ks = Ks.clone()
                Ks[:, 0, :] *= width / cam.width
                Ks[:, 1, :] *= height / cam.height

            rgb, _, info = render(self.params, viewmats, Ks, width, height, sh_deg, self.backend, bg)
            if not self.mcmc:
                self.strategy.step_pre_backward(self.params, self.optimizers, self.strategy_state, step, info)

            pred = rgb[0]
            l1 = F.l1_loss(pred, gt)
            ssim_val = ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
            loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - ssim_val)
            if self.mcmc:
                loss = loss + cfg.mcmc_opacity_reg * torch.sigmoid(self.params["opacities"]).mean()
                loss = loss + cfg.mcmc_scale_reg * torch.exp(self.params["scales"]).mean()
            if cfg.flatten_weight > 0:
                thin = torch.exp(self.params["scales"]).min(dim=1).values
                loss = loss + cfg.flatten_weight * thin.mean() / self.scene.scene_scale
            loss.backward()

            if self.selective:
                visible = (info["radii"] > 0).all(dim=-1).any(dim=0)  # [N]
                for opt in self.optimizers.values():
                    opt.step(visible)
                    opt.zero_grad(set_to_none=True)
            else:
                for opt in self.optimizers.values():
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            self.means_scheduler.step()
            if self.mcmc:
                lr = self.means_scheduler.get_last_lr()[0]
                self.strategy.step_post_backward(self.params, self.optimizers, self.strategy_state, step, info, lr=lr)
            else:
                self.strategy.step_post_backward(
                    self.params, self.optimizers, self.strategy_state, step, info, packed=False
                )
                if step % cfg.refine_every == 0:
                    self._enforce_budget()

            if step % cfg.log_every == 0:
                bar.set_postfix(loss=f"{loss.item():.4f}", n=len(self.params["means"]), sh=sh_deg)
            if cfg.early_stop and len(self.testset) and step > stop_after and step % cfg.eval_every == 0:
                cur = self.quick_psnr()
                if cur > best_psnr + cfg.early_stop_delta:
                    best_psnr, bad_evals = cur, 0
                else:
                    bad_evals += 1
                logger.info("step %d: test PSNR %.2f dB (best %.2f)", step, cur, best_psnr)
                if bad_evals >= cfg.early_stop_patience:
                    logger.info("Early stopping at step %d", step)
                    self.stopped_at = step
                    break
            if cfg.save_every and step > 0 and step % cfg.save_every == 0:
                save_ply(self.params, self.output_dir / f"splat_{step}.ply")

        train_time = time.time() - start
        ply = save_ply(self.params, self.output_dir / "splat.ply")
        metrics = self.evaluate()
        metrics.update(
            num_gaussians=len(self.params["means"]), train_seconds=train_time,
            steps=self.stopped_at if self.stopped_at is not None else cfg.iterations,
        )
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "config": asdict(cfg),
                    "backend": self.backend,
                    "colmap_to_scene_transform": self.scene.transform.tolist(),
                },
                f,
                indent=2,
            )
        logger.info("Saved %s | %s", ply, metrics)
        return TrainResult(
            self.params, ply, metrics, cfg.sh_degree, self.backend,
            self.trainset.cameras, self.testset.cameras,
        )

    @torch.no_grad()
    def evaluate(self, save_renders: bool = True) -> dict[str, float | str]:
        """Compute PSNR / SSIM on held-out images (or training images if none).

        Args:
            save_renders: Save side-by-side ``ground truth | render`` images to
                ``output_dir/renders``.

        Returns:
            Dictionary with ``psnr``, ``ssim`` and the split that was evaluated.
        """
        dataset, split = (self.testset, "test") if len(self.testset) else (self.trainset, "train")
        indices = range(len(dataset)) if split == "test" else range(0, len(dataset), max(1, len(dataset) // 8))
        psnrs, ssims = [], []
        for i in indices:
            cam, gt = dataset[i]
            gt = gt.to(self.device)
            pred = render_camera(self.params, cam, self.cfg.sh_degree, self.backend)
            psnrs.append(psnr(pred, gt))
            ssims.append(ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).item())
            if save_renders:
                save_image(torch.cat([gt, pred], dim=1), self.output_dir / "renders" / f"{split}_{Path(cam.name).stem}.jpg")
        return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)), "eval_split": split}
