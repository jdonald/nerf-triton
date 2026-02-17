"""Triton kernel implementations for NeRF volume rendering and positional encoding.

Provides GPU (triton) and CPU (triton-cpu / interpreter) accelerated kernels for:
- Volume rendering (alpha compositing along rays) — fuses ~10 PyTorch ops
- Positional encoding (Fourier feature computation) — fuses sin/cos loop

Backend options:
- "pytorch": Pure PyTorch reference implementation (default)
- "triton": Triton kernels on CUDA GPU
- "triton-cpu": Triton kernels on CPU (via TRITON_INTERPRET=1 or triton-cpu backend)
- "auto": Automatically select best available backend
"""

import os

import torch

_triton_available = None


def is_triton_available() -> bool:
    """Check if the triton package can be imported."""
    global _triton_available
    if _triton_available is None:
        try:
            import triton
            import triton.language as tl
            _triton_available = True
        except ImportError:
            _triton_available = False
    return _triton_available


def is_triton_cpu_available() -> bool:
    """Check if Triton can run on CPU (interpreter mode or triton-cpu backend)."""
    if not is_triton_available():
        return False
    # CPU execution requires either TRITON_INTERPRET=1 or a native CPU backend
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        return True
    # Check for native CPU backend
    try:
        import triton.backends
        backends_dir = os.path.dirname(triton.backends.__file__)
        return os.path.isdir(os.path.join(backends_dir, "cpu"))
    except Exception:
        return False


def is_triton_gpu_available() -> bool:
    """Check if Triton can run on GPU (requires CUDA)."""
    if not is_triton_available():
        return False
    return torch.cuda.is_available()


def resolve_backend(backend: str, device: torch.device = None) -> str:
    """Resolve 'auto' backend to a concrete backend based on availability.

    Args:
        backend: One of "pytorch", "triton", "triton-cpu", "auto"
        device: Target device (used for 'auto' resolution)

    Returns:
        Resolved backend string
    """
    if backend == "auto":
        if device is not None and device.type == "cuda" and is_triton_gpu_available():
            return "triton"
        if is_triton_cpu_available():
            return "triton-cpu"
        return "pytorch"
    return backend


# ---------------------------------------------------------------------------
# Triton kernel definitions (lazily compiled on first use)
# ---------------------------------------------------------------------------

_volume_render_fwd_kernel = None
_positional_encoding_fwd_kernel = None


def _ensure_kernels():
    """Import triton and define JIT kernels. Called once on first use."""
    global _volume_render_fwd_kernel, _positional_encoding_fwd_kernel
    if _volume_render_fwd_kernel is not None:
        return

    import triton
    import triton.language as tl

    # ----- Volume rendering forward kernel -----
    @triton.jit
    def volume_render_fwd_kernel(
        raw_ptr,               # (N, S, 4) network output [r, g, b, sigma]
        t_vals_ptr,            # (N, S) parametric distances
        rays_d_ptr,            # (N, 3) ray directions
        rgb_out_ptr,           # (N, 3) output RGB
        depth_out_ptr,         # (N,) output depth
        weights_out_ptr,       # (N, S) output weights
        alpha_out_ptr,         # (N, S) saved alpha (for backward)
        transmittance_out_ptr, # (N, S) saved transmittance (for backward)
        dists_out_ptr,         # (N, S) saved dists (for backward)
        N,                     # number of rays
        S: tl.constexpr,      # samples per ray (compile-time constant)
        WHITE_BG: tl.constexpr,
    ):
        """One program per ray. Sequential accumulation over S samples."""
        ray_idx = tl.program_id(0)
        if ray_idx >= N:
            return

        # Ray direction magnitude for distance scaling
        dx = tl.load(rays_d_ptr + ray_idx * 3 + 0)
        dy = tl.load(rays_d_ptr + ray_idx * 3 + 1)
        dz = tl.load(rays_d_ptr + ray_idx * 3 + 2)
        ray_norm = tl.sqrt(dx * dx + dy * dy + dz * dz)

        # Accumulation state
        T = 1.0          # transmittance
        acc_r = 0.0
        acc_g = 0.0
        acc_b = 0.0
        acc_depth = 0.0
        acc_weight = 0.0

        for s in range(S):
            # Load raw [r, g, b, sigma] for this sample
            raw_base = ray_idx * S * 4 + s * 4
            r = tl.load(raw_ptr + raw_base + 0)
            g = tl.load(raw_ptr + raw_base + 1)
            b = tl.load(raw_ptr + raw_base + 2)
            sigma = tl.load(raw_ptr + raw_base + 3)

            # Load parametric distance
            t_idx = ray_idx * S + s
            t_val = tl.load(t_vals_ptr + t_idx)

            # Distance to next sample (last sample extends to infinity)
            t_next = tl.load(
                t_vals_ptr + t_idx + 1,
                mask=(s < S - 1),
                other=t_val + 1e10,
            )
            dist = (t_next - t_val) * ray_norm

            # Alpha from density: alpha = 1 - exp(-relu(sigma) * dist)
            sigma_relu = tl.maximum(sigma, 0.0)
            alpha = 1.0 - tl.exp(-sigma_relu * dist)

            # Compositing weight
            w = T * alpha

            # Accumulate color and depth
            acc_r += w * r
            acc_g += w * g
            acc_b += w * b
            acc_depth += w * t_val
            acc_weight += w

            # Save intermediates for backward pass
            tl.store(weights_out_ptr + t_idx, w)
            tl.store(alpha_out_ptr + t_idx, alpha)
            tl.store(transmittance_out_ptr + t_idx, T)
            tl.store(dists_out_ptr + t_idx, dist)

            # Update transmittance for next sample
            T = T * (1.0 - alpha + 1e-10)

        # White background compositing
        if WHITE_BG:
            acc_r += 1.0 - acc_weight
            acc_g += 1.0 - acc_weight
            acc_b += 1.0 - acc_weight

        # Store final outputs
        tl.store(rgb_out_ptr + ray_idx * 3 + 0, acc_r)
        tl.store(rgb_out_ptr + ray_idx * 3 + 1, acc_g)
        tl.store(rgb_out_ptr + ray_idx * 3 + 2, acc_b)
        tl.store(depth_out_ptr + ray_idx, acc_depth)

    # ----- Positional encoding forward kernel -----
    @triton.jit
    def positional_encoding_fwd_kernel(
        x_ptr,        # (N, D) input coordinates
        out_ptr,      # (N, OUT_DIM) output encoded coordinates
        freq_ptr,     # (L,) frequency bands
        N,            # batch size
        D: tl.constexpr,       # input dimension
        L: tl.constexpr,       # number of frequencies
        OUT_DIM: tl.constexpr, # output dimension = D + 2*L*D
        BLOCK_N: tl.constexpr, # elements per program
    ):
        """Encode positions using Fourier features. One program per block of N."""
        pid = tl.program_id(0)
        n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = n_offsets < N

        for d in range(D):
            # Load input value
            val = tl.load(x_ptr + n_offsets * D + d, mask=mask, other=0.0)

            # Copy raw input to output
            tl.store(out_ptr + n_offsets * OUT_DIM + d, val, mask=mask)

            # Compute sin/cos for each frequency band
            for l_idx in range(L):
                freq = tl.load(freq_ptr + l_idx)
                angle = freq * 3.141592653589793 * val
                sin_val = tl.sin(angle)
                cos_val = tl.cos(angle)

                # Output layout: [x, sin(f0*x), cos(f0*x), sin(f1*x), cos(f1*x), ...]
                # sin index: D + 2*l*D + d
                # cos index: D + (2*l+1)*D + d
                sin_idx = D + 2 * l_idx * D + d
                cos_idx = D + (2 * l_idx + 1) * D + d
                tl.store(out_ptr + n_offsets * OUT_DIM + sin_idx, sin_val, mask=mask)
                tl.store(out_ptr + n_offsets * OUT_DIM + cos_idx, cos_val, mask=mask)

    _volume_render_fwd_kernel = volume_render_fwd_kernel
    _positional_encoding_fwd_kernel = positional_encoding_fwd_kernel


# ---------------------------------------------------------------------------
# autograd.Function wrappers for training support
# ---------------------------------------------------------------------------

def _pytorch_volume_render(raw, t_vals, rays_d, white_background):
    """Pure PyTorch volume rendering (used for backward pass)."""
    dists = t_vals[:, 1:] - t_vals[:, :-1]
    dists = torch.cat([dists, torch.full_like(dists[:, :1], 1e10)], dim=-1)
    dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)

    rgb = raw[..., :3]
    sigma = raw[..., 3]

    alpha = 1.0 - torch.exp(-torch.relu(sigma) * dists)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[:, :-1]

    weights = alpha * transmittance
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * t_vals, dim=-1)

    if white_background:
        acc_map = torch.sum(weights, dim=-1)
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, depth_map, weights


class _TritonVolumeRenderFunction(torch.autograd.Function):
    """Autograd wrapper: Triton forward, PyTorch backward."""

    @staticmethod
    def forward(ctx, raw, t_vals, rays_d, white_background):
        _ensure_kernels()

        N, S, _ = raw.shape
        device = raw.device
        dtype = raw.dtype

        # Ensure contiguous
        raw_c = raw.contiguous()
        t_vals_c = t_vals.contiguous()
        rays_d_c = rays_d.contiguous()

        # Allocate outputs
        rgb_out = torch.empty(N, 3, device=device, dtype=dtype)
        depth_out = torch.empty(N, device=device, dtype=dtype)
        weights_out = torch.empty(N, S, device=device, dtype=dtype)

        # Allocate intermediates for backward
        alpha_out = torch.empty(N, S, device=device, dtype=dtype)
        transmittance_out = torch.empty(N, S, device=device, dtype=dtype)
        dists_out = torch.empty(N, S, device=device, dtype=dtype)

        # Launch kernel: one program per ray
        grid = (N,)
        _volume_render_fwd_kernel[grid](
            raw_c, t_vals_c, rays_d_c,
            rgb_out, depth_out, weights_out,
            alpha_out, transmittance_out, dists_out,
            N, S=S, WHITE_BG=white_background,
        )

        # Save for backward
        ctx.save_for_backward(raw, t_vals, rays_d)
        ctx.white_background = white_background

        return rgb_out, depth_out, weights_out

    @staticmethod
    def backward(ctx, grad_rgb, grad_depth, grad_weights):
        raw, t_vals, rays_d = ctx.saved_tensors
        white_background = ctx.white_background

        # Recompute forward in PyTorch for autograd backward
        with torch.enable_grad():
            raw_g = raw.detach().requires_grad_(True)
            t_vals_g = t_vals.detach().requires_grad_(True)
            rgb_map, depth_map, weights = _pytorch_volume_render(
                raw_g, t_vals_g, rays_d, white_background
            )

            # Accumulate gradients from all outputs
            outputs = []
            grads = []
            if grad_rgb is not None:
                outputs.append(rgb_map)
                grads.append(grad_rgb)
            if grad_depth is not None:
                outputs.append(depth_map)
                grads.append(grad_depth)

            if outputs:
                torch.autograd.backward(outputs, grads)

        return raw_g.grad, t_vals_g.grad, None, None


class _TritonPositionalEncodingFunction(torch.autograd.Function):
    """Autograd wrapper for Triton positional encoding."""

    @staticmethod
    def forward(ctx, x, freq_bands, num_frequencies, input_dim):
        _ensure_kernels()

        N = x.shape[0]
        D = input_dim
        L = num_frequencies
        OUT_DIM = D + D * 2 * L
        device = x.device
        dtype = x.dtype

        x_c = x.contiguous()
        freq_c = freq_bands.contiguous()

        out = torch.empty(N, OUT_DIM, device=device, dtype=dtype)

        BLOCK_N = 128
        grid = ((N + BLOCK_N - 1) // BLOCK_N,)

        _positional_encoding_fwd_kernel[grid](
            x_c, out, freq_c,
            N, D=D, L=L, OUT_DIM=OUT_DIM, BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(x, freq_bands)
        ctx.num_frequencies = num_frequencies
        ctx.input_dim = input_dim

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, freq_bands = ctx.saved_tensors
        D = ctx.input_dim
        L = ctx.num_frequencies

        # Gradient of positional encoding w.r.t. input x
        # output = [x, sin(f0*pi*x), cos(f0*pi*x), ...]
        # d(sin(f*pi*x))/dx = f*pi*cos(f*pi*x)
        # d(cos(f*pi*x))/dx = -f*pi*sin(f*pi*x)
        grad_x = grad_output[:, :D].clone()  # gradient from identity part

        for l_idx in range(L):
            freq = freq_bands[l_idx]
            angle = freq * torch.pi * x

            sin_grad_idx_start = D + 2 * l_idx * D
            cos_grad_idx_start = D + (2 * l_idx + 1) * D

            grad_sin = grad_output[:, sin_grad_idx_start:sin_grad_idx_start + D]
            grad_cos = grad_output[:, cos_grad_idx_start:cos_grad_idx_start + D]

            # Chain rule
            grad_x += grad_sin * (freq * torch.pi * torch.cos(angle))
            grad_x += grad_cos * (-freq * torch.pi * torch.sin(angle))

        return grad_x, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def triton_volume_render(
    raw: torch.Tensor,
    t_vals: torch.Tensor,
    rays_d: torch.Tensor,
    white_background: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Volume rendering using Triton kernels.

    Drop-in replacement for the PyTorch volume_render function.
    Supports autograd for training (backward via PyTorch recomputation).

    Args:
        raw: (N, S, 4) raw network output [r, g, b, sigma]
        t_vals: (N, S) parametric distances along rays
        rays_d: (N, 3) ray directions
        white_background: If True, composite over white background

    Returns:
        rgb_map: (N, 3) rendered RGB colors per ray
        depth_map: (N,) expected depth per ray
        weights: (N, S) sample weights for hierarchical sampling
    """
    return _TritonVolumeRenderFunction.apply(raw, t_vals, rays_d, white_background)


def triton_positional_encode(
    x: torch.Tensor,
    freq_bands: torch.Tensor,
    num_frequencies: int,
    input_dim: int,
) -> torch.Tensor:
    """Positional encoding using Triton kernels.

    Args:
        x: (N, D) input coordinates
        freq_bands: (L,) precomputed frequency bands
        num_frequencies: number of frequency bands L
        input_dim: input dimension D

    Returns:
        (N, D + D*2*L) encoded coordinates
    """
    return _TritonPositionalEncodingFunction.apply(
        x, freq_bands, num_frequencies, input_dim
    )
