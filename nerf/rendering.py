"""Volume rendering for NeRF — supports PyTorch and Triton backends.

This module implements the alpha compositing accumulation along each ray.
The volume_render function dispatches to either the pure-PyTorch reference
implementation or fused Triton kernels based on the ``backend`` parameter.
"""

import torch

from nerf.model import NeRFModel
from nerf.sampling import sample_stratified


def volume_render(
    raw: torch.Tensor,
    t_vals: torch.Tensor,
    rays_d: torch.Tensor,
    white_background: bool = True,
    backend: str = "pytorch",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render colors from raw NeRF output along rays.

    Implements the volume rendering equation:
        C(r) = sum_i T_i * alpha_i * c_i
        T_i  = prod_{j<i} (1 - alpha_j)
        alpha_i = 1 - exp(-sigma_i * delta_i)

    Args:
        raw: (N, S, 4) raw network output [r, g, b, sigma]
        t_vals: (N, S) parametric distances along rays
        rays_d: (N, 3) ray directions (used for distance computation)
        white_background: If True, composite over white background
        backend: "pytorch", "triton", or "triton-cpu"

    Returns:
        rgb_map: (N, 3) rendered RGB colors per ray
        depth_map: (N,) expected depth per ray
        weights: (N, S) sample weights for hierarchical sampling
    """
    if backend in ("triton", "triton-cpu"):
        from nerf.triton_kernels import triton_volume_render
        return triton_volume_render(raw, t_vals, rays_d, white_background)

    # --- PyTorch reference implementation ---
    # Distances between adjacent samples
    dists = t_vals[:, 1:] - t_vals[:, :-1]  # (N, S-1)
    # Last interval extends to infinity (use large value)
    dists = torch.cat([dists, torch.full_like(dists[:, :1], 1e10)], dim=-1)  # (N, S)

    # Scale by ray direction magnitude
    dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)

    # Extract RGB and density
    rgb = raw[..., :3]  # (N, S, 3)
    sigma = raw[..., 3]  # (N, S)

    # Alpha from density: alpha = 1 - exp(-sigma * delta)
    alpha = 1.0 - torch.exp(-torch.relu(sigma) * dists)  # (N, S)

    # Transmittance: T_i = prod_{j<i} (1 - alpha_j)
    # Use cumulative product with exclusive prefix (shift right, prepend 1)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[:, :-1]  # (N, S)

    # Weights for compositing
    weights = alpha * transmittance  # (N, S)

    # Weighted sum of colors
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # (N, 3)

    # Expected depth
    depth_map = torch.sum(weights * t_vals, dim=-1)  # (N,)

    # White background
    if white_background:
        acc_map = torch.sum(weights, dim=-1)  # (N,)
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, depth_map, weights


def render_rays(
    model: NeRFModel,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    perturb: bool = True,
    white_background: bool = True,
    backend: str = "pytorch",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full render pipeline: sample points, query model, composite.

    Args:
        model: NeRF model
        rays_o: (N, 3) ray origins
        rays_d: (N, 3) ray directions
        near: Near clipping distance
        far: Far clipping distance
        num_samples: Samples per ray
        perturb: Jitter samples during training
        white_background: Composite over white
        backend: "pytorch", "triton", or "triton-cpu"

    Returns:
        rgb_map: (N, 3) rendered colors
        depth_map: (N,) rendered depths
        weights: (N, S) sample weights
    """
    # Sample points along rays
    points, t_vals = sample_stratified(
        rays_o, rays_d, near, far, num_samples, perturb=perturb
    )

    # Flatten for model forward pass
    N, S, _ = points.shape
    flat_points = points.reshape(-1, 3)

    # Expand directions to match points
    dirs = rays_d[:, None, :].expand_as(points).reshape(-1, 3)
    dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-8)

    # Query model
    raw = model(flat_points, dirs)  # (N*S, 4)
    raw = raw.reshape(N, S, 4)

    # Volume render
    return volume_render(
        raw, t_vals, rays_d, white_background=white_background, backend=backend
    )


def render_image(
    model: NeRFModel,
    c2w: torch.Tensor,
    height: int,
    width: int,
    focal: float,
    near: float,
    far: float,
    num_samples: int,
    chunk_size: int = 4096,
    white_background: bool = True,
    backend: str = "pytorch",
) -> torch.Tensor:
    """Render a full image from a camera pose.

    Args:
        model: Trained NeRF model
        c2w: (4, 4) camera-to-world matrix
        height: Image height
        width: Image width
        focal: Focal length
        near: Near plane
        far: Far plane
        num_samples: Samples per ray
        chunk_size: Number of rays to process at once
        white_background: White background flag
        backend: "pytorch", "triton", or "triton-cpu"

    Returns:
        (H, W, 3) rendered image tensor in [0, 1]
    """
    from nerf.rays import get_rays

    rays_o, rays_d = get_rays(height, width, focal, c2w)

    all_rgb = []
    for i in range(0, rays_o.shape[0], chunk_size):
        chunk_o = rays_o[i : i + chunk_size]
        chunk_d = rays_d[i : i + chunk_size]
        rgb, _, _ = render_rays(
            model,
            chunk_o,
            chunk_d,
            near,
            far,
            num_samples,
            perturb=False,
            white_background=white_background,
            backend=backend,
        )
        all_rgb.append(rgb)

    rgb_map = torch.cat(all_rgb, dim=0)
    return rgb_map.reshape(height, width, 3)
