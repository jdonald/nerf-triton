"""Tests for synthetic training data generation."""

import json
import os

import numpy as np
import pytest
from PIL import Image

from data.render_views import (
    generate_camera_poses,
    generate_dataset,
    project_vertices,
    render_scene,
)
from data.generate_flower import generate_flower
from nerf.rays import look_at


class TestCameraPoses:
    def test_correct_count(self):
        poses = generate_camera_poses(20)
        assert len(poses) == 20

    def test_pose_shape(self):
        poses = generate_camera_poses(5)
        for p in poses:
            assert p.shape == (4, 4)

    def test_poses_on_hemisphere(self):
        """Cameras should be approximately at the specified radius."""
        radius = 3.0
        poses = generate_camera_poses(50, radius=radius)
        for p in poses:
            cam_pos = p[:3, 3]
            dist = np.linalg.norm(cam_pos)
            np.testing.assert_allclose(dist, radius, atol=0.01)

    def test_cameras_above_equator(self):
        """All cameras should be in the upper hemisphere (y > 0)."""
        poses = generate_camera_poses(50, radius=3.0)
        for p in poses:
            assert p[1, 3] > -0.1  # allow small tolerance


class TestProjectVertices:
    def test_output_shape(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        c2w = look_at(np.array([0.0, 0.0, 3.0]))
        screen = project_vertices(verts, c2w, focal=50.0, height=100, width=100)
        assert screen.shape == (3, 3)

    def test_center_projects_to_center(self):
        """Origin should project to the image center."""
        c2w = look_at(np.array([0.0, 0.0, 3.0]))
        verts = np.array([[0.0, 0.0, 0.0]])
        screen = project_vertices(verts, c2w, focal=50.0, height=100, width=100)
        np.testing.assert_allclose(screen[0, :2], [50.0, 50.0], atol=1.0)


class TestRenderScene:
    def test_output_shape(self):
        v, n, c, f = generate_flower()
        c2w = look_at(np.array([0.0, 1.0, 3.0]))
        img = render_scene(v, f, c, n, c2w, focal=50.0, height=32, width=32)
        assert img.shape == (32, 32, 3)

    def test_output_range(self):
        v, n, c, f = generate_flower()
        c2w = look_at(np.array([0.0, 1.0, 3.0]))
        img = render_scene(v, f, c, n, c2w, focal=50.0, height=32, width=32)
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_not_all_background(self):
        """Scene should have some non-background pixels."""
        v, n, c, f = generate_flower()
        c2w = look_at(np.array([0.0, 0.5, 3.0]))
        img = render_scene(v, f, c, n, c2w, focal=80.0, height=64, width=64)
        # Check that some pixels differ from white background
        white = np.array([1.0, 1.0, 1.0])
        non_bg = np.any(np.abs(img - white) > 0.1, axis=-1)
        assert non_bg.sum() > 10  # at least some flower pixels


class TestGenerateDataset:
    def test_creates_files(self, tmp_path):
        data_dir = str(tmp_path / "data")
        generate_dataset(
            output_dir=data_dir,
            n_views=5,
            image_size=16,
            radius=3.0,
        )

        # Check transforms.json exists
        transforms_path = os.path.join(data_dir, "transforms.json")
        assert os.path.exists(transforms_path)

        with open(transforms_path) as f:
            meta = json.load(f)
        assert len(meta["frames"]) == 5
        assert "focal" in meta
        assert "camera_angle_x" in meta

        # Check images exist
        images_dir = os.path.join(data_dir, "images")
        assert os.path.exists(images_dir)
        images = os.listdir(images_dir)
        assert len(images) == 5

    def test_images_correct_size(self, tmp_path):
        data_dir = str(tmp_path / "data")
        generate_dataset(output_dir=data_dir, n_views=2, image_size=32)

        img = Image.open(os.path.join(data_dir, "images", "r_000.png"))
        assert img.size == (32, 32)

    def test_transforms_matrices_valid(self, tmp_path):
        data_dir = str(tmp_path / "data")
        generate_dataset(output_dir=data_dir, n_views=3, image_size=16)

        with open(os.path.join(data_dir, "transforms.json")) as f:
            meta = json.load(f)

        for frame in meta["frames"]:
            mat = np.array(frame["transform_matrix"])
            assert mat.shape == (4, 4)
            # Should be a valid rotation + translation
            R = mat[:3, :3]
            det = np.linalg.det(R)
            np.testing.assert_allclose(abs(det), 1.0, atol=1e-5)
