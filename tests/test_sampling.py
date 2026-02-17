"""Tests for ray sampling strategies."""

import torch
import pytest

from nerf.sampling import sample_stratified, sample_hierarchical


class TestSampleStratified:
    def test_output_shapes(self):
        N, S = 10, 32
        rays_o = torch.randn(N, 3)
        rays_d = torch.randn(N, 3)
        points, t_vals = sample_stratified(rays_o, rays_d, near=2.0, far=6.0,
                                           num_samples=S)
        assert points.shape == (N, S, 3)
        assert t_vals.shape == (N, S)

    def test_t_vals_sorted(self):
        """t values should be monotonically increasing along each ray."""
        rays_o = torch.randn(20, 3)
        rays_d = torch.randn(20, 3)
        _, t_vals = sample_stratified(rays_o, rays_d, near=1.0, far=5.0,
                                      num_samples=64, perturb=True)
        diffs = t_vals[:, 1:] - t_vals[:, :-1]
        assert (diffs >= -1e-6).all(), "t_vals should be non-decreasing"

    def test_t_vals_within_bounds(self):
        """t values should be within [near, far]."""
        near, far = 2.0, 6.0
        rays_o = torch.randn(10, 3)
        rays_d = torch.randn(10, 3)
        _, t_vals = sample_stratified(rays_o, rays_d, near=near, far=far,
                                      num_samples=32, perturb=True)
        assert t_vals.min() >= near - 0.01
        assert t_vals.max() <= far + 0.2  # small margin for jitter

    def test_no_perturb_is_uniform(self):
        """Without perturbation, t values should be exactly uniform."""
        near, far = 2.0, 6.0
        rays_o = torch.zeros(1, 3)
        rays_d = torch.tensor([[0.0, 0.0, -1.0]])
        _, t_vals = sample_stratified(rays_o, rays_d, near=near, far=far,
                                      num_samples=10, perturb=False)
        expected = torch.linspace(near, far, 10).unsqueeze(0)
        torch.testing.assert_close(t_vals, expected)

    def test_points_on_ray(self):
        """Sampled points should lie on the ray: p = o + t*d."""
        rays_o = torch.tensor([[1.0, 0.0, 0.0]])
        rays_d = torch.tensor([[0.0, 0.0, -1.0]])
        points, t_vals = sample_stratified(rays_o, rays_d, near=1.0, far=5.0,
                                           num_samples=10, perturb=False)
        for s in range(10):
            expected = rays_o + t_vals[0, s] * rays_d
            torch.testing.assert_close(points[0, s], expected[0], atol=1e-5, rtol=1e-5)


class TestSampleHierarchical:
    def test_output_shapes(self):
        N, S_coarse = 5, 16
        rays_o = torch.randn(N, 3)
        rays_d = torch.randn(N, 3)
        t_vals = torch.linspace(2.0, 6.0, S_coarse).unsqueeze(0).expand(N, -1)
        weights = torch.rand(N, S_coarse)

        points, t_combined = sample_hierarchical(
            rays_o, rays_d, t_vals, weights,
            num_importance=32, perturb=False,
        )
        assert t_combined.shape == (N, S_coarse + 32)
        assert points.shape == (N, S_coarse + 32, 3)

    def test_combined_sorted(self):
        """Combined t values should be sorted."""
        N = 5
        t_vals = torch.linspace(2.0, 6.0, 16).unsqueeze(0).expand(N, -1)
        weights = torch.rand(N, 16)
        rays_o = torch.randn(N, 3)
        rays_d = torch.randn(N, 3)

        _, t_combined = sample_hierarchical(
            rays_o, rays_d, t_vals, weights,
            num_importance=16, perturb=False,
        )
        diffs = t_combined[:, 1:] - t_combined[:, :-1]
        assert (diffs >= -1e-5).all()
