"""FPS-style camera with mouse look, WASD movement, and zoom."""

import math

import numpy as np


class FPSCamera:
    """First-person camera for interactive scene navigation.

    Controls:
        Mouse: Look around (yaw/pitch)
        W/S: Move forward/backward
        A/D: Move left/right
        +/-: Zoom in/out (adjusts FOV)
    """

    def __init__(
        self,
        position: np.ndarray = None,
        yaw: float = -90.0,
        pitch: float = 0.0,
        fov: float = 60.0,
        move_speed: float = 2.5,
        mouse_sensitivity: float = 0.1,
    ):
        self.position = np.array(position if position is not None else [0.0, 0.5, 3.0], dtype=np.float64)
        self.yaw = yaw  # degrees
        self.pitch = pitch  # degrees
        self.fov = fov  # degrees
        self.move_speed = move_speed
        self.mouse_sensitivity = mouse_sensitivity

        # Clamp limits
        self.min_pitch = -89.0
        self.max_pitch = 89.0
        self.min_fov = 10.0
        self.max_fov = 120.0

        self._update_vectors()

    def _update_vectors(self):
        """Recompute forward, right, up vectors from yaw/pitch."""
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)

        self.forward = np.array([
            math.cos(pitch_rad) * math.cos(yaw_rad),
            math.sin(pitch_rad),
            math.cos(pitch_rad) * math.sin(yaw_rad),
        ])
        self.forward /= np.linalg.norm(self.forward)

        world_up = np.array([0.0, 1.0, 0.0])
        self.right = np.cross(self.forward, world_up)
        norm = np.linalg.norm(self.right)
        if norm > 1e-6:
            self.right /= norm
        else:
            self.right = np.array([1.0, 0.0, 0.0])

        self.up = np.cross(self.right, self.forward)

    def process_mouse(self, dx: float, dy: float):
        """Process mouse movement for look-around.

        Args:
            dx: Horizontal mouse delta (pixels)
            dy: Vertical mouse delta (pixels)
        """
        self.yaw += dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity  # invert Y
        self.pitch = max(self.min_pitch, min(self.max_pitch, self.pitch))
        self._update_vectors()

    def process_keyboard(self, keys: dict, dt: float):
        """Process keyboard input for movement.

        Args:
            keys: dict with booleans for 'w', 'a', 's', 'd'
            dt: time delta in seconds
        """
        velocity = self.move_speed * dt

        if keys.get("w"):
            self.position += self.forward * velocity
        if keys.get("s"):
            self.position -= self.forward * velocity
        if keys.get("a"):
            self.position -= self.right * velocity
        if keys.get("d"):
            self.position += self.right * velocity

    def zoom(self, amount: float):
        """Adjust FOV for zoom effect.

        Args:
            amount: Positive = zoom in (decrease FOV), negative = zoom out
        """
        self.fov -= amount
        self.fov = max(self.min_fov, min(self.max_fov, self.fov))

    def get_view_matrix(self) -> np.ndarray:
        """Get the 4x4 view matrix (world-to-camera)."""
        target = self.position + self.forward
        return self._look_at_matrix(self.position, target, self.up)

    def get_c2w_matrix(self) -> np.ndarray:
        """Get the 4x4 camera-to-world matrix for NeRF compatibility."""
        c2w = np.eye(4)
        c2w[:3, 0] = self.right
        c2w[:3, 1] = self.up
        c2w[:3, 2] = -self.forward  # OpenGL: camera looks down -Z
        c2w[:3, 3] = self.position
        return c2w

    @staticmethod
    def _look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Compute a look-at view matrix."""
        f = target - eye
        f = f / np.linalg.norm(f)
        r = np.cross(f, up)
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)

        view = np.eye(4)
        view[0, :3] = r
        view[1, :3] = u
        view[2, :3] = -f
        view[0, 3] = -np.dot(r, eye)
        view[1, 3] = -np.dot(u, eye)
        view[2, 3] = np.dot(f, eye)
        return view
