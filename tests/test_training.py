"""Tests for NeRF training pipeline."""

import os
import tempfile

import numpy as np
import torch
import pytest

from nerf.config import TinyNeRFConfig
from nerf.model import NeRFModel
from nerf.train import train, load_model


@pytest.fixture
def tiny_dataset(tmp_path):
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


class TestTraining:
    def test_loss_decreases(self, tiny_dataset):
        """Training should reduce loss over epochs."""
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

        output_dir = os.path.join(os.path.dirname(tiny_dataset), "checkpoints")
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

    def test_checkpoint_saved(self, tiny_dataset):
        """Training should save a checkpoint file."""
        config = TinyNeRFConfig()
        config.num_epochs = 2
        config.batch_size = 64
        config.num_samples = 8
        config.num_layers = 2
        config.hidden_dim = 32
        config.num_freq_position = 2
        config.num_freq_direction = 1

        output_dir = os.path.join(os.path.dirname(tiny_dataset), "checkpoints")
        train(
            config,
            data_dir=tiny_dataset,
            output_dir=output_dir,
            device=torch.device("cpu"),
            verbose=False,
        )

        ckpt_path = os.path.join(output_dir, "model.pt")
        assert os.path.exists(ckpt_path)

    def test_load_model(self, tiny_dataset):
        """Should be able to load a saved model and run inference."""
        config = TinyNeRFConfig()
        config.num_epochs = 2
        config.batch_size = 64
        config.num_samples = 8
        config.num_layers = 2
        config.hidden_dim = 32
        config.num_freq_position = 2
        config.num_freq_direction = 1

        output_dir = os.path.join(os.path.dirname(tiny_dataset), "checkpoints")
        train(
            config,
            data_dir=tiny_dataset,
            output_dir=output_dir,
            device=torch.device("cpu"),
            verbose=False,
        )

        ckpt_path = os.path.join(output_dir, "model.pt")
        model, meta = load_model(ckpt_path, torch.device("cpu"))

        assert isinstance(model, NeRFModel)
        assert "focal" in meta
        assert "image_height" in meta
        assert "near" in meta

        # Run inference
        pos = torch.randn(5, 3)
        dirs = torch.randn(5, 3)
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        with torch.no_grad():
            out = model(pos, dirs)
        assert out.shape == (5, 4)
