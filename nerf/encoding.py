"""Positional encoding using Fourier features."""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Maps input coordinates to higher-dimensional space using sin/cos.

    gamma(p) = [p, sin(2^0 * pi * p), cos(2^0 * pi * p), ...,
                   sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p)]

    Input dim d -> output dim d + d * 2 * L
    """

    def __init__(self, num_frequencies: int, input_dim: int = 3):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.input_dim = input_dim
        self.output_dim = input_dim + input_dim * 2 * num_frequencies
        # Precompute frequency bands: 2^0, 2^1, ..., 2^(L-1)
        freq_bands = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input coordinates.

        Args:
            x: (N, d) input coordinates

        Returns:
            (N, d + d*2*L) encoded coordinates
        """
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * torch.pi * x))
            encoded.append(torch.cos(freq * torch.pi * x))
        return torch.cat(encoded, dim=-1)
