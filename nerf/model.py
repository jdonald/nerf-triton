"""NeRF MLP network architecture."""

import torch
import torch.nn as nn

from nerf.encoding import PositionalEncoding


class NeRFModel(nn.Module):
    """NeRF multi-layer perceptron.

    Architecture:
    - 8 FC layers (256 channels) with ReLU, skip connection at layer 4
    - Position input -> density (sigma) + feature vector
    - Direction input -> RGB color via one additional layer
    """

    def __init__(
        self,
        num_layers: int = 8,
        hidden_dim: int = 256,
        skip_layer: int = 4,
        num_freq_position: int = 10,
        num_freq_direction: int = 4,
    ):
        super().__init__()
        self.skip_layer = skip_layer

        # Positional encodings
        self.pos_encoder = PositionalEncoding(num_freq_position, input_dim=3)
        self.dir_encoder = PositionalEncoding(num_freq_direction, input_dim=3)

        pos_dim = self.pos_encoder.output_dim
        dir_dim = self.dir_encoder.output_dim

        # Position MLP layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(pos_dim, hidden_dim))
        for i in range(1, num_layers):
            if i == skip_layer:
                # Skip connection: concatenate original input
                self.layers.append(nn.Linear(hidden_dim + pos_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Density output (no activation — raw sigma)
        self.density_head = nn.Linear(hidden_dim, 1)

        # Color branch: feature + direction -> RGB
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        self.color_layer = nn.Linear(hidden_dim + dir_dim, hidden_dim // 2)
        self.rgb_head = nn.Linear(hidden_dim // 2, 3)

        self.relu = nn.ReLU()

    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            positions: (N, 3) 3D positions
            directions: (N, 3) viewing directions (unit vectors)

        Returns:
            (N, 4) tensor of [R, G, B, sigma] where RGB in [0,1]
        """
        # Encode inputs
        pos_encoded = self.pos_encoder(positions)
        dir_encoded = self.dir_encoder(directions)

        # Position MLP with skip connection
        h = pos_encoded
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                h = torch.cat([h, pos_encoded], dim=-1)
            h = self.relu(layer(h))

        # Density
        sigma = self.density_head(h)

        # Color
        feature = self.feature_layer(h)
        color_input = torch.cat([feature, dir_encoded], dim=-1)
        color = self.relu(self.color_layer(color_input))
        rgb = torch.sigmoid(self.rgb_head(color))

        return torch.cat([rgb, sigma], dim=-1)
