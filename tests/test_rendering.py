"""Tests for volume rendering."""

import torch
import pytest

from nerf.rendering import volume_render, render_rays
from nerf.model import NeRFModel


class TestVolumeRender:
    def test_output_shapes(self):
        N, S = 10, 32
        raw = torch.randn(N, S, 4)
        raw[..., :3] = torch.sigmoid(raw[..., :3])  # RGB in [0,1]
        t_vals = torch.linspace(2.0, 6.0, S).unsqueeze(0).expand(N, -1)
        rays_d = torch.randn(N, 3)

        rgb, depth, weights = volume_render(raw, t_vals, rays_d)
        assert rgb.shape == (N, 3)
        assert depth.shape == (N,)
        assert weights.shape == (N, S)

    def test_constant_color_volume(self):
        """A constant-density, constant-color volume should render as that color."""
        N, S = 5, 64
        target_color = torch.tensor([0.8, 0.2, 0.5])

        raw = torch.zeros(N, S, 4)
        raw[..., :3] = target_color  # constant color
        raw[..., 3] = 10.0  # high density (opaque)

        t_vals = torch.linspace(2.0, 6.0, S).unsqueeze(0).expand(N, -1)
        rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(N, -1)

        rgb, depth, weights = volume_render(raw, t_vals, rays_d, white_background=False)

        # Should closely approximate the target color
        for i in range(N):
            torch.testing.assert_close(rgb[i], target_color, atol=0.05, rtol=0.1)

    def test_empty_volume(self):
        """Zero density should render as background (white)."""
        N, S = 3, 32
        raw = torch.zeros(N, S, 4)
        raw[..., :3] = 0.5  # some color, but zero density
        raw[..., 3] = 0.0  # zero density

        t_vals = torch.linspace(2.0, 6.0, S).unsqueeze(0).expand(N, -1)
        rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(N, -1)

        rgb, depth, weights = volume_render(raw, t_vals, rays_d, white_background=True)

        # Should be white (background)
        expected = torch.ones(N, 3)
        torch.testing.assert_close(rgb, expected, atol=0.01, rtol=0.01)

    def test_weights_sum_to_one_or_less(self):
        """Weights along each ray should sum to at most 1."""
        N, S = 10, 32
        raw = torch.randn(N, S, 4)
        raw[..., 3] = torch.abs(raw[..., 3])  # positive density
        t_vals = torch.linspace(2.0, 6.0, S).unsqueeze(0).expand(N, -1)
        rays_d = torch.randn(N, 3)

        _, _, weights = volume_render(raw, t_vals, rays_d)
        weight_sums = weights.sum(dim=-1)
        assert (weight_sums <= 1.0 + 1e-5).all()

    def test_depth_within_bounds(self):
        """Expected depth should be between near and far."""
        N, S = 10, 64
        near, far = 2.0, 6.0
        raw = torch.randn(N, S, 4)
        raw[..., 3] = torch.abs(raw[..., 3]) + 0.1  # positive density
        t_vals = torch.linspace(near, far, S).unsqueeze(0).expand(N, -1)
        rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(N, -1)

        _, depth, weights = volume_render(raw, t_vals, rays_d)
        # Depth is a weighted average of t_vals, so should be in [near, far]
        assert (depth >= near - 0.1).all()
        assert (depth <= far + 0.1).all()


class TestRenderRays:
    def test_output_shapes(self):
        model = NeRFModel(num_layers=2, hidden_dim=32,
                          num_freq_position=2, num_freq_direction=1)
        N = 16
        rays_o = torch.randn(N, 3)
        rays_d = torch.randn(N, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        rgb, depth, weights = render_rays(
            model, rays_o, rays_d,
            near=2.0, far=6.0, num_samples=8,
        )
        assert rgb.shape == (N, 3)
        assert depth.shape == (N,)

    def test_rgb_range(self):
        """Rendered colors should be in [0, 1]."""
        model = NeRFModel(num_layers=2, hidden_dim=32,
                          num_freq_position=2, num_freq_direction=1)
        rays_o = torch.randn(20, 3)
        rays_d = torch.randn(20, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        rgb, _, _ = render_rays(
            model, rays_o, rays_d,
            near=2.0, far=6.0, num_samples=8,
        )
        assert rgb.min() >= -0.01  # small tolerance
        assert rgb.max() <= 1.01
