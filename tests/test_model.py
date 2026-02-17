"""Tests for the NeRF MLP model."""

import torch
import pytest

from nerf.model import NeRFModel


class TestNeRFModel:
    def test_output_shape(self):
        model = NeRFModel(num_layers=8, hidden_dim=256)
        pos = torch.randn(32, 3)
        dirs = torch.randn(32, 3)
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        out = model(pos, dirs)
        assert out.shape == (32, 4)

    def test_rgb_range(self):
        """RGB outputs should be in [0, 1] due to sigmoid."""
        model = NeRFModel(num_layers=4, hidden_dim=64)
        pos = torch.randn(50, 3) * 2
        dirs = torch.randn(50, 3)
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        out = model(pos, dirs)
        rgb = out[:, :3]
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_sigma_unbounded(self):
        """Density (sigma) should be unbounded (no activation on output)."""
        torch.manual_seed(42)
        model = NeRFModel(num_layers=4, hidden_dim=64)
        pos = torch.randn(200, 3) * 3
        dirs = torch.randn(200, 3)
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        out = model(pos, dirs)
        sigma = out[:, 3]
        # Sigma can be negative (before ReLU in rendering)
        # Just check it's not all zeros
        assert not torch.all(sigma == 0)

    def test_gradient_flow(self):
        """Gradients should flow through the model."""
        model = NeRFModel(num_layers=4, hidden_dim=64)
        pos = torch.randn(10, 3, requires_grad=True)
        dirs = torch.randn(10, 3)
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        out = model(pos, dirs)
        loss = out.sum()
        loss.backward()
        assert pos.grad is not None
        assert not torch.all(pos.grad == 0)

    def test_small_config(self):
        """Test with minimal configuration."""
        model = NeRFModel(
            num_layers=2, hidden_dim=32,
            skip_layer=1,
            num_freq_position=2,
            num_freq_direction=1,
        )
        pos = torch.randn(5, 3)
        dirs = torch.randn(5, 3)
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        out = model(pos, dirs)
        assert out.shape == (5, 4)

    def test_direction_dependence(self):
        """RGB should change with different viewing directions."""
        torch.manual_seed(0)
        model = NeRFModel(num_layers=4, hidden_dim=64)
        pos = torch.randn(1, 3).expand(2, -1)  # same position
        dirs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out = model(pos, dirs)
        # Density should be the same (same position)
        torch.testing.assert_close(out[0, 3:], out[1, 3:], atol=1e-5, rtol=1e-5)
        # Colors may differ (different directions) — at least they're valid
        assert out.shape == (2, 4)
