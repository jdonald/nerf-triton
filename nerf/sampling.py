"""Ray sampling strategies for volume rendering."""

import torch


def sample_stratified(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stratified sampling along rays.

    Divides [near, far] into num_samples uniform bins and samples one point
    per bin, with optional jitter for regularization during training.

    Args:
        rays_o: (N, 3) ray origins
        rays_d: (N, 3) ray directions
        near: Near clipping distance
        far: Far clipping distance
        num_samples: Number of samples per ray
        perturb: If True, add uniform jitter within each bin

    Returns:
        points: (N, num_samples, 3) sampled 3D points
        t_vals: (N, num_samples) parametric distances along rays
    """
    device = rays_o.device
    N = rays_o.shape[0]

    # Uniform bin edges
    t_vals = torch.linspace(near, far, num_samples, device=device)
    t_vals = t_vals.unsqueeze(0).expand(N, num_samples)  # (N, S)

    if perturb:
        # Add jitter within each bin
        bin_size = (far - near) / num_samples
        t_vals = t_vals + torch.rand_like(t_vals) * bin_size

    # Compute 3D points: o + t*d
    points = rays_o[:, None, :] + t_vals[:, :, None] * rays_d[:, None, :]

    return points, t_vals


def sample_hierarchical(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    t_vals: torch.Tensor,
    weights: torch.Tensor,
    num_importance: int,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hierarchical (importance) sampling based on coarse weights.

    Samples additional points in regions with high opacity, guided by the
    probability distribution from coarse network weights.

    Args:
        rays_o: (N, 3) ray origins
        rays_d: (N, 3) ray directions
        t_vals: (N, S_coarse) coarse sample distances
        weights: (N, S_coarse) coarse sample weights
        num_importance: Number of additional importance samples
        perturb: If True, add noise to CDF inversion

    Returns:
        points: (N, S_coarse + num_importance, 3) all sampled points (sorted)
        t_vals_combined: (N, S_coarse + num_importance) combined sorted t values
    """
    device = rays_o.device
    N = t_vals.shape[0]

    # Construct PDF from weights (add small epsilon for stability)
    weights_pad = weights + 1e-5
    pdf = weights_pad / torch.sum(weights_pad, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)  # (N, S+1)

    # Sample from CDF via inverse transform
    if perturb:
        u = torch.rand(N, num_importance, device=device)
    else:
        u = torch.linspace(0.0, 1.0, num_importance, device=device)
        u = u.unsqueeze(0).expand(N, num_importance)

    # Invert CDF
    indices = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    below = torch.clamp(indices - 1, min=0)
    above = torch.clamp(indices, max=cdf.shape[-1] - 1)

    cdf_below = torch.gather(cdf, 1, below)
    cdf_above = torch.gather(cdf, 1, above)
    t_below = torch.gather(t_vals, 1, torch.clamp(below, max=t_vals.shape[-1] - 1))
    t_above = torch.gather(t_vals, 1, torch.clamp(above - 1, max=t_vals.shape[-1] - 1))

    # Linear interpolation
    denom = cdf_above - cdf_below
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t_new = t_below + (u - cdf_below) / denom * (t_above - t_below)

    # Combine and sort
    t_vals_combined, _ = torch.sort(torch.cat([t_vals, t_new], dim=-1), dim=-1)

    # Compute 3D points
    points = rays_o[:, None, :] + t_vals_combined[:, :, None] * rays_d[:, None, :]

    return points, t_vals_combined
