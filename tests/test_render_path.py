"""End-to-end test: train a tiny NeRF and render a short camera path."""

import os
import tempfile

import numpy as np
import torch
import pytest
from PIL import Image

from nerf.config import TinyNeRFConfig
from nerf.train import train
from nerf.rendering import render_image
from nerf.train import load_model
from viewer.app import run_headless
from playback.render_path import render_camera_path


@pytest.fixture
def trained_model_and_dataset(tmp_path):
    """Train a tiny NeRF model for testing."""
    from data.render_views import generate_dataset

    data_dir = str(tmp_path / "data")
    generate_dataset(
        output_dir=data_dir,
        n_views=4,
        image_size=16,
        radius=3.0,
    )

    config = TinyNeRFConfig()
    config.num_epochs = 3
    config.batch_size = 64
    config.num_samples = 8
    config.num_layers = 2
    config.hidden_dim = 32
    config.num_freq_position = 2
    config.num_freq_direction = 1

    output_dir = str(tmp_path / "checkpoints")
    model, losses = train(
        config,
        data_dir=data_dir,
        output_dir=output_dir,
        device=torch.device("cpu"),
        verbose=False,
    )
    ckpt_path = os.path.join(output_dir, "model.pt")
    return ckpt_path, data_dir, model


class TestRenderPath:
    def test_render_single_image(self, trained_model_and_dataset):
        """Render a single image from the trained model."""
        ckpt_path, data_dir, _ = trained_model_and_dataset
        model, meta = load_model(ckpt_path, torch.device("cpu"))

        c2w = torch.eye(4)
        c2w[2, 3] = 3.0  # camera at z=3

        with torch.no_grad():
            img = render_image(
                model, c2w,
                height=meta["image_height"],
                width=meta["image_width"],
                focal=meta["focal"],
                near=meta["near"],
                far=meta["far"],
                num_samples=meta["num_samples"],
                chunk_size=256,
            )

        assert img.shape == (meta["image_height"], meta["image_width"], 3)
        assert img.min() >= -0.01
        assert img.max() <= 1.01

    def test_headless_viewer_produces_path(self, tmp_path):
        """Headless viewer should produce a valid camera path file."""
        output_path = str(tmp_path / "test_path.json")
        poses = run_headless(num_frames=10, output_path=output_path)

        assert len(poses) == 10
        assert os.path.exists(output_path)

        # Load and verify
        from viewer.recorder import PathRecorder
        frames = PathRecorder.load(output_path)
        assert len(frames) == 10
        for frame in frames:
            mat = np.array(frame["transform_matrix"])
            assert mat.shape == (4, 4)

    def test_end_to_end_render_path(self, trained_model_and_dataset, tmp_path):
        """Full pipeline: train -> record path -> render frames."""
        ckpt_path, data_dir, _ = trained_model_and_dataset

        # Generate a camera path
        path_file = str(tmp_path / "camera_path.json")
        run_headless(num_frames=3, output_path=path_file)

        # Render through NeRF
        output_dir = str(tmp_path / "frames")
        output_paths = render_camera_path(
            checkpoint_path=ckpt_path,
            path_file=path_file,
            output_dir=output_dir,
            device=torch.device("cpu"),
            verbose=False,
        )

        assert len(output_paths) == 3
        for p in output_paths:
            assert os.path.exists(p)
            img = Image.open(p)
            assert img.size[0] > 0
            assert img.size[1] > 0

    def test_rendered_frames_have_correct_dimensions(self, trained_model_and_dataset, tmp_path):
        """Rendered frames should match expected image dimensions."""
        ckpt_path, _, _ = trained_model_and_dataset

        path_file = str(tmp_path / "camera_path.json")
        run_headless(num_frames=2, output_path=path_file)

        output_dir = str(tmp_path / "frames")
        output_paths = render_camera_path(
            checkpoint_path=ckpt_path,
            path_file=path_file,
            output_dir=output_dir,
            device=torch.device("cpu"),
            verbose=False,
        )

        _, meta = load_model(ckpt_path, torch.device("cpu"))
        for p in output_paths:
            img = Image.open(p)
            assert img.size == (meta["image_width"], meta["image_height"])
