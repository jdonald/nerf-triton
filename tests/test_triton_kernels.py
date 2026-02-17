"""Tests for Triton kernel implementations.

Tests cover:
- Numerical equivalence between PyTorch and Triton volume rendering
- Numerical equivalence between PyTorch and Triton positional encoding
- Gradient correctness through Triton autograd wrappers
- End-to-end training pipeline with Triton backend
- Backend selection and availability detection
"""

import os
import pytest
import torch

from nerf.triton_kernels import (
    is_triton_available,
    is_triton_cpu_available,
    is_triton_gpu_available,
    resolve_backend,
)

# All Triton tests require TRITON_INTERPRET=1 on CPU (no GPU in CI)
requires_triton = pytest.mark.skipif(
    not is_triton_available(),
    reason="Triton not installed",
)

requires_triton_cpu = pytest.mark.skipif(
    not is_triton_cpu_available(),
    reason="Triton CPU backend not available (set TRITON_INTERPRET=1)",
)


class TestBackendDetection:
    def test_triton_import(self):
        assert is_triton_available() is True

    def test_resolve_pytorch(self):
        assert resolve_backend("pytorch") == "pytorch"

    def test_resolve_triton_cpu(self):
        assert resolve_backend("triton-cpu") == "triton-cpu"

    def test_resolve_auto_no_cuda(self):
        device = torch.device("cpu")
        result = resolve_backend("auto", device)
        if is_triton_cpu_available():
            assert result == "triton-cpu"
        else:
            assert result == "pytorch"

    @requires_triton
    def test_resolve_auto_with_cuda(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            result = resolve_backend("auto", device)
            assert result == "triton"


class TestTritonVolumeRender:
    """Test Triton volume rendering kernel against PyTorch reference."""

    @requires_triton_cpu
    def test_output_shapes(self):
        from nerf.triton_kernels import triton_volume_render

        N, S = 10, 32
        raw = torch.randn(N, S, 4)
        raw[..., :3] = torch.sigmoid(raw[..., :3])
        t_vals = torch.linspace(2.0, 6.0, S).unsqueeze(0).expand(N, -1).contiguous()
        rays_d = torch.randn(N, 3)

        rgb, depth, weights = triton_volume_render(raw, t_vals, rays_d)
        assert rgb.shape == (N, 3)
        assert depth.shape == (N,)
        assert weights.shape == (N, S)

    @requires_triton_cpu
    def test_numerical_equivalence(self):
        """Triton and PyTorch volume render should produce matching results."""
        from nerf.rendering import volume_render

        torch.manual_seed(42)
        N, S = 16, 32
        raw = torch.randn(N, S, 4)
        raw[..., :3] = torch.sigmoid(raw[..., :3])
        t_vals = torch.linspace(2.0, 6.0, S).unsqueeze(0).expand(N, -1).contiguous()
        rays_d = torch.randn(N, 3)

        # PyTorch reference
        rgb_pt, depth_pt, weights_pt = volume_render(
            raw, t_vals, rays_d, white_background=True, backend="pytorch"
        )
        # Triton
        rgb_tr, depth_tr, weights_tr = volume_render(
            raw, t_vals, rays_d, white_background=True, backend="triton-cpu"
        )

        torch.testing.assert_close(rgb_tr, rgb_pt, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(depth_tr, depth_pt, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(weights_tr, weights_pt, atol=1e-4, rtol=1e-4)

    @requires_triton_cpu
    def test_numerical_equivalence_no_white_bg(self):
        """Test equivalence with white_background=False."""
        from nerf.rendering import volume_render

        torch.manual_seed(123)
        N, S = 8, 16
        raw = torch.randn(N, S, 4)
        t_vals = torch.linspace(2.0, 6.0, S).unsqueeze(0).expand(N, -1).contiguous()
        rays_d = torch.randn(N, 3)

        rgb_pt, depth_pt, weights_pt = volume_render(
            raw, t_vals, rays_d, white_background=False, backend="pytorch"
        )
        rgb_tr, depth_tr, weights_tr = volume_render(
            raw, t_vals, rays_d, white_background=False, backend="triton-cpu"
        )

        torch.testing.assert_close(rgb_tr, rgb_pt, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(depth_tr, depth_pt, atol=1e-4, rtol=1e-4)

    @requires_triton_cpu
    def test_constant_color_volume(self):
        """High-density constant-color volume should render as that color."""
        from nerf.triton_kernels import triton_volume_render

        N, S = 5, 64
        target_color = torch.tensor([0.8, 0.2, 0.5])

        raw = torch.zeros(N, S, 4)
        raw[..., :3] = target_color
        raw[..., 3] = 10.0  # high density

        t_vals = torch.linspace(2.0, 6.0, S).unsqueeze(0).expand(N, -1).contiguous()
        rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(N, -1)

        rgb, _, _ = triton_volume_render(raw, t_vals, rays_d, white_background=False)

        for i in range(N):
            torch.testing.assert_close(rgb[i], target_color, atol=0.05, rtol=0.1)

    @requires_triton_cpu
    def test_empty_volume(self):
        """Zero density should render as white background."""
        from nerf.triton_kernels import triton_volume_render

        N, S = 3, 32
        raw = torch.zeros(N, S, 4)
        raw[..., :3] = 0.5
        raw[..., 3] = 0.0

        t_vals = torch.linspace(2.0, 6.0, S).unsqueeze(0).expand(N, -1).contiguous()
        rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(N, -1)

        rgb, _, _ = triton_volume_render(raw, t_vals, rays_d, white_background=True)

        expected = torch.ones(N, 3)
        torch.testing.assert_close(rgb, expected, atol=0.01, rtol=0.01)

    @requires_triton_cpu
    def test_weights_sum_to_one_or_less(self):
        from nerf.triton_kernels import triton_volume_render

        N, S = 10, 32
        raw = torch.randn(N, S, 4)
        raw[..., 3] = torch.abs(raw[..., 3])
        t_vals = torch.linspace(2.0, 6.0, S).unsqueeze(0).expand(N, -1).contiguous()
        rays_d = torch.randn(N, 3)

        _, _, weights = triton_volume_render(raw, t_vals, rays_d)
        weight_sums = weights.sum(dim=-1)
        assert (weight_sums <= 1.0 + 1e-5).all()

    @requires_triton_cpu
    def test_gradient_flows(self):
        """Backward pass should produce valid gradients through Triton forward."""
        from nerf.triton_kernels import triton_volume_render

        N, S = 8, 16
        raw = torch.randn(N, S, 4, requires_grad=True)
        t_vals = torch.linspace(2.0, 6.0, S).unsqueeze(0).expand(N, -1).contiguous()
        rays_d = torch.randn(N, 3)

        rgb, depth, weights = triton_volume_render(raw, t_vals, rays_d)

        loss = rgb.sum() + depth.sum()
        loss.backward()

        assert raw.grad is not None
        assert raw.grad.shape == raw.shape
        assert torch.isfinite(raw.grad).all()

    @requires_triton_cpu
    def test_gradient_numerical_check(self):
        """Triton gradients should match numerical (finite-difference) gradients."""
        from nerf.triton_kernels import triton_volume_render

        N, S = 4, 8
        raw = torch.randn(N, S, 4, requires_grad=True, dtype=torch.float64)
        t_vals = torch.linspace(2.0, 6.0, S, dtype=torch.float64).unsqueeze(0).expand(N, -1).contiguous()
        rays_d = torch.randn(N, 3, dtype=torch.float64)

        def func(raw_input):
            rgb, depth, _ = triton_volume_render(raw_input, t_vals, rays_d, True)
            return rgb.sum() + depth.sum()

        passed = torch.autograd.gradcheck(
            func, (raw,), eps=1e-5, atol=1e-3, rtol=1e-3, nondet_tol=1e-3,
        )
        assert passed

    @requires_triton_cpu
    def test_various_sample_counts(self):
        """Kernel should work for different numbers of samples per ray."""
        from nerf.rendering import volume_render

        for S in [4, 8, 16, 32, 64]:
            N = 4
            raw = torch.randn(N, S, 4)
            t_vals = torch.linspace(2.0, 6.0, S).unsqueeze(0).expand(N, -1).contiguous()
            rays_d = torch.randn(N, 3)

            rgb_pt, _, _ = volume_render(raw, t_vals, rays_d, backend="pytorch")
            rgb_tr, _, _ = volume_render(raw, t_vals, rays_d, backend="triton-cpu")
            torch.testing.assert_close(rgb_tr, rgb_pt, atol=1e-4, rtol=1e-4)


class TestTritonPositionalEncoding:
    """Test Triton positional encoding kernel against PyTorch reference."""

    @requires_triton_cpu
    def test_output_shape(self):
        from nerf.encoding import PositionalEncoding

        enc = PositionalEncoding(num_frequencies=10, input_dim=3, backend="triton-cpu")
        x = torch.randn(100, 3)
        out = enc(x)
        assert out.shape == (100, 63)

    @requires_triton_cpu
    def test_numerical_equivalence(self):
        """Triton and PyTorch positional encoding should produce matching results."""
        from nerf.encoding import PositionalEncoding

        torch.manual_seed(42)
        x = torch.randn(50, 3)

        enc_pt = PositionalEncoding(num_frequencies=6, input_dim=3, backend="pytorch")
        enc_tr = PositionalEncoding(num_frequencies=6, input_dim=3, backend="triton-cpu")

        out_pt = enc_pt(x)
        out_tr = enc_tr(x)

        torch.testing.assert_close(out_tr, out_pt, atol=1e-5, rtol=1e-5)

    @requires_triton_cpu
    def test_identity_prefix(self):
        """First d columns should be the original input."""
        from nerf.encoding import PositionalEncoding

        enc = PositionalEncoding(num_frequencies=4, input_dim=3, backend="triton-cpu")
        x = torch.randn(10, 3)
        out = enc(x)
        torch.testing.assert_close(out[:, :3], x)

    @requires_triton_cpu
    def test_sin_cos_range(self):
        """Sin/cos components should be in [-1, 1]."""
        from nerf.encoding import PositionalEncoding

        enc = PositionalEncoding(num_frequencies=10, input_dim=3, backend="triton-cpu")
        x = torch.randn(200, 3) * 5
        out = enc(x)
        sincos = out[:, 3:]
        assert sincos.min() >= -1.0 - 1e-6
        assert sincos.max() <= 1.0 + 1e-6

    @requires_triton_cpu
    def test_different_frequencies(self):
        """Should match PyTorch for various frequency counts."""
        from nerf.encoding import PositionalEncoding

        torch.manual_seed(7)
        x = torch.randn(20, 3)

        for L in [1, 2, 4, 6, 10]:
            enc_pt = PositionalEncoding(num_frequencies=L, input_dim=3, backend="pytorch")
            enc_tr = PositionalEncoding(num_frequencies=L, input_dim=3, backend="triton-cpu")
            out_pt = enc_pt(x)
            out_tr = enc_tr(x)
            torch.testing.assert_close(out_tr, out_pt, atol=1e-5, rtol=1e-5)

    @requires_triton_cpu
    def test_different_input_dims(self):
        """Should work with non-3D input."""
        from nerf.encoding import PositionalEncoding

        torch.manual_seed(99)
        for D in [2, 3, 5]:
            x = torch.randn(15, D)
            enc_pt = PositionalEncoding(num_frequencies=3, input_dim=D, backend="pytorch")
            enc_tr = PositionalEncoding(num_frequencies=3, input_dim=D, backend="triton-cpu")
            out_pt = enc_pt(x)
            out_tr = enc_tr(x)
            torch.testing.assert_close(out_tr, out_pt, atol=1e-5, rtol=1e-5)

    @requires_triton_cpu
    def test_gradient_flows(self):
        """Backward pass should produce valid gradients."""
        from nerf.encoding import PositionalEncoding

        enc = PositionalEncoding(num_frequencies=4, input_dim=3, backend="triton-cpu")
        x = torch.randn(10, 3, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()

    @requires_triton_cpu
    def test_gradient_matches_pytorch(self):
        """Triton and PyTorch gradients should match."""
        from nerf.encoding import PositionalEncoding

        torch.manual_seed(42)
        x_data = torch.randn(10, 3)

        # PyTorch gradients
        x_pt = x_data.clone().requires_grad_(True)
        enc_pt = PositionalEncoding(num_frequencies=4, input_dim=3, backend="pytorch")
        out_pt = enc_pt(x_pt)
        out_pt.sum().backward()

        # Triton gradients
        x_tr = x_data.clone().requires_grad_(True)
        enc_tr = PositionalEncoding(num_frequencies=4, input_dim=3, backend="triton-cpu")
        out_tr = enc_tr(x_tr)
        out_tr.sum().backward()

        torch.testing.assert_close(x_tr.grad, x_pt.grad, atol=1e-4, rtol=1e-4)


class TestTritonRenderRays:
    """Test the full render_rays pipeline with Triton backend."""

    @requires_triton_cpu
    def test_output_shapes(self):
        from nerf.model import NeRFModel
        from nerf.rendering import render_rays

        model = NeRFModel(
            num_layers=2, hidden_dim=32,
            num_freq_position=2, num_freq_direction=1,
            backend="triton-cpu",
        )
        N = 16
        rays_o = torch.randn(N, 3)
        rays_d = torch.randn(N, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        rgb, depth, weights = render_rays(
            model, rays_o, rays_d,
            near=2.0, far=6.0, num_samples=8,
            backend="triton-cpu",
        )
        assert rgb.shape == (N, 3)
        assert depth.shape == (N,)

    @requires_triton_cpu
    def test_rgb_range(self):
        from nerf.model import NeRFModel
        from nerf.rendering import render_rays

        model = NeRFModel(
            num_layers=2, hidden_dim=32,
            num_freq_position=2, num_freq_direction=1,
            backend="triton-cpu",
        )
        rays_o = torch.randn(20, 3)
        rays_d = torch.randn(20, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        rgb, _, _ = render_rays(
            model, rays_o, rays_d,
            near=2.0, far=6.0, num_samples=8,
            backend="triton-cpu",
        )
        assert rgb.min() >= -0.01
        assert rgb.max() <= 1.01

    @requires_triton_cpu
    def test_render_rays_matches_pytorch(self):
        """Full render_rays should produce close results across backends."""
        from nerf.model import NeRFModel
        from nerf.rendering import render_rays

        torch.manual_seed(42)

        # Use same weights for both models
        model_pt = NeRFModel(
            num_layers=2, hidden_dim=32,
            num_freq_position=2, num_freq_direction=1,
            backend="pytorch",
        )
        model_tr = NeRFModel(
            num_layers=2, hidden_dim=32,
            num_freq_position=2, num_freq_direction=1,
            backend="triton-cpu",
        )
        # Copy weights
        model_tr.load_state_dict(model_pt.state_dict())

        rays_o = torch.randn(8, 3)
        rays_d = torch.randn(8, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        # No perturbation so results are deterministic
        rgb_pt, depth_pt, _ = render_rays(
            model_pt, rays_o, rays_d,
            near=2.0, far=6.0, num_samples=8,
            perturb=False, backend="pytorch",
        )
        rgb_tr, depth_tr, _ = render_rays(
            model_tr, rays_o, rays_d,
            near=2.0, far=6.0, num_samples=8,
            perturb=False, backend="triton-cpu",
        )

        torch.testing.assert_close(rgb_tr, rgb_pt, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(depth_tr, depth_pt, atol=1e-4, rtol=1e-4)


class TestTritonEndToEnd:
    """End-to-end training pipeline with Triton backend."""

    @pytest.fixture
    def tiny_dataset(self, tmp_path):
        """Generate a minimal training dataset for testing."""
        from data.render_views import generate_dataset

        data_dir = str(tmp_path / "data")
        generate_dataset(
            output_dir=data_dir,
            n_views=4,
            image_size=16,
            radius=3.0,
        )
        return data_dir

    @requires_triton_cpu
    def test_training_loss_decreases(self, tiny_dataset, tmp_path):
        """Training with Triton backend should reduce loss over epochs."""
        from nerf.config import TinyNeRFConfig
        from nerf.train import train

        config = TinyNeRFConfig()
        config.num_epochs = 10
        config.batch_size = 128
        config.num_samples = 8
        config.num_layers = 2
        config.hidden_dim = 32
        config.num_freq_position = 2
        config.num_freq_direction = 1
        config.image_size = 16
        config.learning_rate = 1e-3
        config.backend = "triton-cpu"

        output_dir = str(tmp_path / "checkpoints")
        model, losses = train(
            config,
            data_dir=tiny_dataset,
            output_dir=output_dir,
            device=torch.device("cpu"),
            verbose=False,
        )

        assert len(losses) == 10
        # Average of last 3 epochs should be lower than average of first 3
        avg_first = sum(losses[:3]) / 3
        avg_last = sum(losses[-3:]) / 3
        assert avg_last < avg_first, f"Loss should trend down: {losses}"

    @requires_triton_cpu
    def test_checkpoint_roundtrip(self, tiny_dataset, tmp_path):
        """Train with Triton, save checkpoint, load and verify inference."""
        from nerf.config import TinyNeRFConfig
        from nerf.train import train, load_model

        config = TinyNeRFConfig()
        config.num_epochs = 2
        config.batch_size = 64
        config.num_samples = 8
        config.num_layers = 2
        config.hidden_dim = 32
        config.num_freq_position = 2
        config.num_freq_direction = 1
        config.backend = "triton-cpu"

        output_dir = str(tmp_path / "checkpoints")
        train(
            config,
            data_dir=tiny_dataset,
            output_dir=output_dir,
            device=torch.device("cpu"),
            verbose=False,
        )

        ckpt_path = os.path.join(output_dir, "model.pt")
        assert os.path.exists(ckpt_path)

        model, meta = load_model(ckpt_path, torch.device("cpu"))
        pos = torch.randn(5, 3)
        dirs = torch.randn(5, 3)
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        with torch.no_grad():
            out = model(pos, dirs)
        assert out.shape == (5, 4)

    @requires_triton_cpu
    def test_triton_and_pytorch_training_converge(self, tiny_dataset, tmp_path):
        """Both backends should produce models that converge to similar loss levels."""
        from nerf.config import TinyNeRFConfig
        from nerf.train import train

        # Train with PyTorch backend
        config_pt = TinyNeRFConfig()
        config_pt.num_epochs = 5
        config_pt.batch_size = 128
        config_pt.num_samples = 8
        config_pt.num_layers = 2
        config_pt.hidden_dim = 32
        config_pt.num_freq_position = 2
        config_pt.num_freq_direction = 1
        config_pt.learning_rate = 1e-3
        config_pt.backend = "pytorch"

        _, losses_pt = train(
            config_pt,
            data_dir=tiny_dataset,
            output_dir=str(tmp_path / "ckpt_pt"),
            device=torch.device("cpu"),
            verbose=False,
        )

        # Train with Triton backend
        config_tr = TinyNeRFConfig()
        config_tr.num_epochs = 5
        config_tr.batch_size = 128
        config_tr.num_samples = 8
        config_tr.num_layers = 2
        config_tr.hidden_dim = 32
        config_tr.num_freq_position = 2
        config_tr.num_freq_direction = 1
        config_tr.learning_rate = 1e-3
        config_tr.backend = "triton-cpu"

        _, losses_tr = train(
            config_tr,
            data_dir=tiny_dataset,
            output_dir=str(tmp_path / "ckpt_tr"),
            device=torch.device("cpu"),
            verbose=False,
        )

        # Both should have loss < 0.5 (converging) — exact values differ
        # due to random shuffling but both should learn
        assert losses_pt[-1] < 0.5, f"PyTorch final loss too high: {losses_pt[-1]}"
        assert losses_tr[-1] < 0.5, f"Triton final loss too high: {losses_tr[-1]}"


class TestConfigBackend:
    """Test config backend field."""

    def test_default_backend_is_pytorch(self):
        from nerf.config import NeRFConfig
        config = NeRFConfig()
        assert config.backend == "pytorch"

    def test_custom_backend(self):
        from nerf.config import NeRFConfig
        config = NeRFConfig(backend="triton-cpu")
        assert config.backend == "triton-cpu"

    def test_tiny_config_inherits_backend(self):
        from nerf.config import TinyNeRFConfig
        config = TinyNeRFConfig()
        assert config.backend == "pytorch"
        config.backend = "triton-cpu"
        assert config.backend == "triton-cpu"
