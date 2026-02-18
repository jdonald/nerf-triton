"""Tests for procedural flower mesh generation."""

import numpy as np
import pytest

from data.generate_flower import generate_flower, generate_petal, generate_stem, generate_pistil


class TestGeneratePetal:
    def test_returns_four_arrays(self):
        v, n, c, f = generate_petal()
        assert v.ndim == 2 and v.shape[1] == 3
        assert n.ndim == 2 and n.shape[1] == 3
        assert c.ndim == 2 and c.shape[1] == 3
        assert f.ndim == 2 and f.shape[1] == 3

    def test_vertex_normal_color_count_match(self):
        v, n, c, f = generate_petal()
        assert len(v) == len(n) == len(c)

    def test_normals_unit_length(self):
        _, n, _, _ = generate_petal()
        norms = np.linalg.norm(n, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=0.05)

    def test_colors_in_range(self):
        _, _, c, _ = generate_petal()
        assert c.min() >= 0.0
        assert c.max() <= 1.0

    def test_faces_valid_indices(self):
        v, _, _, f = generate_petal()
        assert f.min() >= 0
        assert f.max() < len(v)


class TestGenerateStem:
    def test_returns_four_arrays(self):
        v, n, c, f = generate_stem()
        assert v.shape[1] == 3
        assert n.shape[1] == 3

    def test_stem_extends_downward(self):
        v, _, _, _ = generate_stem()
        assert v[:, 1].min() < 0  # extends below origin


class TestGeneratePistil:
    def test_returns_four_arrays(self):
        v, n, c, f = generate_pistil()
        assert v.shape[1] == 3

    def test_centered_near_origin(self):
        v, _, _, _ = generate_pistil()
        center = v.mean(axis=0)
        assert np.linalg.norm(center) < 0.5  # near origin


class TestGenerateFlower:
    def test_returns_four_arrays(self):
        v, n, c, f = generate_flower(n_petals=8)
        assert v.ndim == 2 and v.shape[1] == 3
        assert n.ndim == 2 and n.shape[1] == 3
        assert c.ndim == 2 and c.shape[1] == 3
        assert f.ndim == 2 and f.shape[1] == 3

    def test_has_many_vertices(self):
        v, _, _, _ = generate_flower(n_petals=8)
        # 8 petals + stem + pistil -> should have many vertices
        assert len(v) > 100

    def test_faces_valid(self):
        v, _, _, f = generate_flower()
        assert f.min() >= 0
        assert f.max() < len(v)

    def test_bounding_box_reasonable(self):
        v, _, _, _ = generate_flower()
        # Flower should fit in a reasonable bounding box
        assert v.min() > -3.0
        assert v.max() < 3.0

    def test_float32_output(self):
        v, n, c, f = generate_flower()
        assert v.dtype == np.float32
        assert n.dtype == np.float32
        assert c.dtype == np.float32
        assert f.dtype == np.int32

    def test_different_petal_counts(self):
        v4, _, _, _ = generate_flower(n_petals=4)
        v12, _, _, _ = generate_flower(n_petals=12)
        # More petals = more vertices
        assert len(v12) > len(v4)
