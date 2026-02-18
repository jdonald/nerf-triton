"""Tests for camera path recording."""

import json
import os
import tempfile

import numpy as np
import pytest

from viewer.recorder import PathRecorder


class TestPathRecorder:
    def test_initial_state(self):
        rec = PathRecorder()
        assert not rec.recording
        assert rec.frame_count == 0

    def test_start_stop(self):
        rec = PathRecorder()
        rec.start()
        assert rec.recording
        rec.stop()
        assert not rec.recording

    def test_toggle(self):
        rec = PathRecorder()
        result = rec.toggle()
        assert result is True
        assert rec.recording
        result = rec.toggle()
        assert result is False
        assert not rec.recording

    def test_capture_while_recording(self):
        rec = PathRecorder()
        rec.start()
        c2w = np.eye(4)
        rec.capture(c2w, fov=60.0)
        rec.capture(c2w, fov=55.0)
        assert rec.frame_count == 2

    def test_capture_while_not_recording(self):
        rec = PathRecorder()
        c2w = np.eye(4)
        rec.capture(c2w, fov=60.0)
        assert rec.frame_count == 0

    def test_save_and_load(self):
        rec = PathRecorder()
        rec.start()

        # Record 10 frames with different poses
        for i in range(10):
            c2w = np.eye(4)
            c2w[0, 3] = float(i)
            rec.capture(c2w, fov=60.0 - i)

        rec.stop()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            rec.save(path)

            # Load back
            frames = PathRecorder.load(path)
            assert len(frames) == 10

            # Verify data roundtrip
            for i, frame in enumerate(frames):
                assert frame["fov"] == 60.0 - i
                mat = np.array(frame["transform_matrix"])
                assert mat.shape == (4, 4)
                np.testing.assert_allclose(mat[0, 3], float(i))
        finally:
            os.unlink(path)

    def test_start_clears_previous(self):
        rec = PathRecorder()
        rec.start()
        rec.capture(np.eye(4), 60.0)
        rec.capture(np.eye(4), 60.0)
        rec.stop()
        assert rec.frame_count == 2

        # Starting again should clear
        rec.start()
        assert rec.frame_count == 0

    def test_save_file_format(self):
        rec = PathRecorder()
        rec.start()
        rec.capture(np.eye(4), 60.0)
        rec.stop()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            rec.save(path)
            with open(path) as f:
                data = json.load(f)
            assert "num_frames" in data
            assert "frames" in data
            assert data["num_frames"] == 1
            assert len(data["frames"]) == 1
            assert "transform_matrix" in data["frames"][0]
            assert "fov" in data["frames"][0]
        finally:
            os.unlink(path)
