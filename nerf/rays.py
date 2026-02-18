"""Ray generation from camera parameters."""

import torch
import numpy as np


def get_rays(
    height: int,
    width: int,
    focal: float,
    c2w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate rays for all pixels in an image.

    Uses the pinhole camera model. Pixel (0,0) is top-left.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        focal: Focal length in pixels
        c2w: (4, 4) camera-to-world transformation matrix

    Returns:
        rays_o: (H*W, 3) ray origins in world space
        rays_d: (H*W, 3) ray directions in world space (not normalized)
    """
    device = c2w.device

    # Pixel grid — center of each pixel
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device),
        indexing="xy",
    )

    # Camera-space directions (OpenGL convention: +X right, +Y up, -Z forward)
    dirs = torch.stack(
        [
            (i - width * 0.5) / focal,
            -(j - height * 0.5) / focal,
            -torch.ones_like(i),
        ],
        dim=-1,
    )  # (H, W, 3)

    # Transform to world space
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)  # (H, W, 3)
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def get_rays_for_poses(
    height: int,
    width: int,
    focal: float,
    poses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate rays for a batch of camera poses.

    Args:
        height: Image height
        width: Image width
        focal: Focal length
        poses: (B, 4, 4) batch of camera-to-world matrices

    Returns:
        rays_o: (B, H*W, 3) ray origins
        rays_d: (B, H*W, 3) ray directions
    """
    all_origins = []
    all_directions = []
    for pose in poses:
        o, d = get_rays(height, width, focal, pose)
        all_origins.append(o)
        all_directions.append(d)
    return torch.stack(all_origins), torch.stack(all_directions)


def look_at(
    cam_pos: np.ndarray,
    target: np.ndarray = None,
    up: np.ndarray = None,
) -> np.ndarray:
    """Compute camera-to-world matrix for a camera looking at a target.

    Args:
        cam_pos: (3,) camera position in world space
        target: (3,) point to look at (default: origin)
        up: (3,) world up vector (default: +Y)

    Returns:
        (4, 4) camera-to-world matrix (OpenGL convention)
    """
    if target is None:
        target = np.array([0.0, 0.0, 0.0])
    if up is None:
        up = np.array([0.0, 1.0, 0.0])

    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm < 1e-6:
        # forward is parallel to up; pick an arbitrary right vector
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / norm

    new_up = np.cross(right, forward)

    # OpenGL convention: camera looks down -Z
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = new_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = cam_pos
    return c2w
