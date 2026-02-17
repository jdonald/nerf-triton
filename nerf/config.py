"""Hyperparameters and configuration for NeRF training and rendering."""

from dataclasses import dataclass, field


@dataclass
class NeRFConfig:
    # Model architecture
    num_layers: int = 8
    hidden_dim: int = 256
    skip_layer: int = 4

    # Positional encoding frequencies
    num_freq_position: int = 10  # L for xyz -> 3 + 3*2*10 = 63 dims
    num_freq_direction: int = 4  # L for dir -> 3 + 3*2*4 = 27 dims

    # Sampling
    num_samples: int = 64  # samples per ray (coarse)
    near: float = 2.0
    far: float = 6.0

    # Training
    learning_rate: float = 5e-4
    num_epochs: int = 50
    batch_size: int = 1024  # rays per batch
    lr_decay_steps: int = 500
    lr_decay_factor: float = 0.1

    # Data
    image_size: int = 100
    num_views: int = 100

    # Rendering
    render_image_size: int = 200
    chunk_size: int = 4096  # rays to process at once during inference


@dataclass
class TinyNeRFConfig(NeRFConfig):
    """Smaller config for testing."""
    num_layers: int = 4
    hidden_dim: int = 64
    num_freq_position: int = 6
    num_freq_direction: int = 2
    num_samples: int = 32
    image_size: int = 50
    num_views: int = 20
    num_epochs: int = 10
    batch_size: int = 512
    render_image_size: int = 50
