"""Tests for positional encoding."""

import torch
import pytest

from nerf.encoding import PositionalEncoding


class TestPositionalEncoding:
    def test_output_shape(self):
        enc = PositionalEncoding(num_frequencies=10, input_dim=3)
        x = torch.randn(100, 3)
        out = enc(x)
        # d + d * 2 * L = 3 + 3 * 2 * 10 = 63
        assert out.shape == (100, 63)

    def test_output_shape_direction(self):
        enc = PositionalEncoding(num_frequencies=4, input_dim=3)
        x = torch.randn(50, 3)
        out = enc(x)
        # 3 + 3 * 2 * 4 = 27
        assert out.shape == (50, 27)

    def test_output_dim_property(self):
        enc = PositionalEncoding(num_frequencies=6, input_dim=3)
        assert enc.output_dim == 3 + 3 * 2 * 6

    def test_identity_prefix(self):
        """First d columns should be the original input."""
        enc = PositionalEncoding(num_frequencies=4, input_dim=3)
        x = torch.randn(10, 3)
        out = enc(x)
        torch.testing.assert_close(out[:, :3], x)

    def test_sin_cos_range(self):
        """Sin/cos components should be in [-1, 1]."""
        enc = PositionalEncoding(num_frequencies=10, input_dim=3)
        x = torch.randn(200, 3) * 5  # larger range inputs
        out = enc(x)
        sincos = out[:, 3:]  # skip identity prefix
        assert sincos.min() >= -1.0
        assert sincos.max() <= 1.0

    def test_deterministic(self):
        """Same input should produce same output."""
        enc = PositionalEncoding(num_frequencies=6, input_dim=3)
        x = torch.randn(10, 3)
        out1 = enc(x)
        out2 = enc(x)
        torch.testing.assert_close(out1, out2)

    def test_batch_independence(self):
        """Encoding of one sample shouldn't depend on others in the batch."""
        enc = PositionalEncoding(num_frequencies=4, input_dim=3)
        x = torch.randn(5, 3)
        out_batch = enc(x)
        out_single = enc(x[2:3])
        torch.testing.assert_close(out_batch[2:3], out_single)

    def test_different_input_dims(self):
        """Should work with non-3D input."""
        enc = PositionalEncoding(num_frequencies=3, input_dim=2)
        x = torch.randn(10, 2)
        out = enc(x)
        assert out.shape == (10, 2 + 2 * 2 * 3)
