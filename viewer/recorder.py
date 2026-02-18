"""Camera path recording and playback."""

import json

import numpy as np


class PathRecorder:
    """Records camera poses over time for later NeRF rendering.

    Usage:
        recorder = PathRecorder()
        recorder.start()
        # Each frame:
        recorder.capture(camera.get_c2w_matrix(), camera.fov)
        recorder.stop()
        recorder.save("camera_path.json")
    """

    def __init__(self):
        self.frames: list[dict] = []
        self.recording = False

    def start(self):
        """Begin recording camera poses."""
        self.frames = []
        self.recording = True

    def stop(self):
        """Stop recording."""
        self.recording = False

    def toggle(self):
        """Toggle recording on/off. Returns new recording state."""
        if self.recording:
            self.stop()
        else:
            self.start()
        return self.recording

    def capture(self, c2w: np.ndarray, fov: float):
        """Record a single frame's camera state.

        Args:
            c2w: (4, 4) camera-to-world matrix
            fov: Field of view in degrees
        """
        if not self.recording:
            return
        self.frames.append({
            "transform_matrix": c2w.tolist(),
            "fov": fov,
        })

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def save(self, path: str):
        """Save recorded path to JSON file.

        Args:
            path: Output file path
        """
        data = {
            "num_frames": len(self.frames),
            "frames": self.frames,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> list[dict]:
        """Load a recorded camera path from JSON.

        Args:
            path: Input file path

        Returns:
            List of frame dicts with 'transform_matrix' and 'fov'
        """
        with open(path) as f:
            data = json.load(f)
        return data["frames"]
