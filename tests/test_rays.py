"""Tests for ray generation."""

import math

import numpy as np
import torch
import pytest

from nerf.rays import get_rays, look_at


class TestGetRays:
    def test_output_shape(self):
        c2w = torch.eye(4)
        rays_o, rays_d = get_rays(10, 15, focal=50.0, c2w=c2w)
        assert rays_o.shape == (150, 3)
        assert rays_d.shape == (150, 3)

    def test_origin_at_camera(self):
        """All ray origins should be at the camera position."""
        c2w = torch.eye(4)
        c2w[0, 3] = 1.0  # camera at (1, 0, 0)
        c2w[1, 3] = 2.0
        c2w[2, 3] = 3.0
        rays_o, _ = get_rays(5, 5, focal=25.0, c2w=c2w)
        for i in range(rays_o.shape[0]):
            torch.testing.assert_close(
                rays_o[i], torch.tensor([1.0, 2.0, 3.0]), atol=1e-5, rtol=1e-5
            )

    def test_center_ray_direction(self):
        """Center pixel should shoot along -Z in camera space (identity pose)."""
        c2w = torch.eye(4)
        H, W = 11, 11
        focal = 100.0
        _, rays_d = get_rays(H, W, focal, c2w)
        # Center pixel is at index (5, 5) -> flat index 5*11+5 = 60
        center_idx = (H // 2) * W + (W // 2)
        d = rays_d[center_idx]
        # Should point in -Z direction (0, 0, -1) approximately
        d_normalized = d / torch.norm(d)
        assert d_normalized[2].item() < -0.99

    def test_rays_diverge(self):
        """Rays at different pixels should point in different directions."""
        c2w = torch.eye(4)
        _, rays_d = get_rays(10, 10, focal=50.0, c2w=c2w)
        # First and last ray should differ
        assert not torch.allclose(rays_d[0], rays_d[-1])


class TestLookAt:
    def test_returns_4x4(self):
        c2w = look_at(np.array([0.0, 0.0, 3.0]))
        assert c2w.shape == (4, 4)

    def test_camera_position(self):
        """Translation column should be the camera position."""
        pos = np.array([1.0, 2.0, 3.0])
        c2w = look_at(pos, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(c2w[:3, 3], pos, atol=1e-10)

    def test_orthonormal_rotation(self):
        """Rotation part should be orthonormal."""
        c2w = look_at(np.array([2.0, 1.0, 3.0]))
        R = c2w[:3, :3]
        # R^T @ R should be identity
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
        # det(R) should be +1
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_looking_at_target(self):
        """Camera should look toward the target (forward = -Z column of c2w)."""
        cam_pos = np.array([0.0, 0.0, 5.0])
        target = np.array([0.0, 0.0, 0.0])
        c2w = look_at(cam_pos, target)
        # -Z column of c2w is the forward direction in world space
        forward = -c2w[:3, 2]
        expected_forward = target - cam_pos
        expected_forward = expected_forward / np.linalg.norm(expected_forward)
        np.testing.assert_allclose(forward, expected_forward, atol=1e-10)
