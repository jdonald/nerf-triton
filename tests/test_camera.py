"""Tests for FPS camera."""

import math

import numpy as np
import pytest

from viewer.camera import FPSCamera


class TestFPSCamera:
    def test_initial_position(self):
        cam = FPSCamera(position=np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(cam.position, [1.0, 2.0, 3.0])

    def test_view_matrix_shape(self):
        cam = FPSCamera()
        view = cam.get_view_matrix()
        assert view.shape == (4, 4)

    def test_c2w_matrix_shape(self):
        cam = FPSCamera()
        c2w = cam.get_c2w_matrix()
        assert c2w.shape == (4, 4)

    def test_c2w_position(self):
        """Camera position should be in the translation column."""
        pos = np.array([1.0, 2.0, 3.0])
        cam = FPSCamera(position=pos)
        c2w = cam.get_c2w_matrix()
        np.testing.assert_allclose(c2w[:3, 3], pos, atol=1e-10)

    def test_c2w_orthonormal(self):
        """Rotation part of c2w should be orthonormal."""
        cam = FPSCamera(yaw=-45.0, pitch=20.0)
        c2w = cam.get_c2w_matrix()
        R = c2w[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_mouse_updates_yaw_pitch(self):
        cam = FPSCamera(yaw=0.0, pitch=0.0)
        cam.process_mouse(100.0, 50.0)
        assert cam.yaw != 0.0
        assert cam.pitch != 0.0

    def test_pitch_clamped(self):
        cam = FPSCamera(pitch=0.0)
        # Try to pitch way beyond limits
        cam.process_mouse(0.0, -10000.0)
        assert cam.pitch <= 89.0
        cam.process_mouse(0.0, 20000.0)
        assert cam.pitch >= -89.0

    def test_wasd_movement(self):
        cam = FPSCamera(position=np.array([0.0, 0.0, 0.0]), yaw=-90.0, pitch=0.0)
        initial_pos = cam.position.copy()

        # Move forward
        cam.process_keyboard({"w": True}, dt=1.0)
        assert not np.allclose(cam.position, initial_pos)

    def test_wasd_directions(self):
        """W and S should move in opposite directions."""
        cam1 = FPSCamera(position=np.array([0.0, 0.0, 0.0]))
        cam2 = FPSCamera(position=np.array([0.0, 0.0, 0.0]))

        cam1.process_keyboard({"w": True}, dt=0.5)
        cam2.process_keyboard({"s": True}, dt=0.5)

        # They should have moved in opposite directions
        dot = np.dot(cam1.position, cam2.position)
        assert dot < 0  # opposite directions from origin

    def test_zoom(self):
        cam = FPSCamera(fov=60.0)
        cam.zoom(10.0)
        assert cam.fov == 50.0
        cam.zoom(-20.0)
        assert cam.fov == 70.0

    def test_zoom_clamped(self):
        cam = FPSCamera(fov=60.0)
        cam.zoom(200.0)  # try extreme zoom in
        assert cam.fov >= cam.min_fov
        cam.zoom(-200.0)  # try extreme zoom out
        assert cam.fov <= cam.max_fov

    def test_forward_vector_unit(self):
        cam = FPSCamera(yaw=45.0, pitch=30.0)
        assert abs(np.linalg.norm(cam.forward) - 1.0) < 1e-10
