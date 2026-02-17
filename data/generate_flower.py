"""Procedural 3D flower mesh generation.

Generates a flower with petals, stem, and pistil center using parametric
equations. Returns vertices, normals, colors, and triangle faces suitable
for both OpenGL rendering and NeRF training data generation.
"""

import numpy as np


def generate_petal(
    n_radial: int = 12,
    n_angular: int = 20,
    petal_length: float = 0.8,
    petal_width: float = 0.35,
    curl_amount: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a single petal as a triangle mesh.

    The petal lies along the +X axis, centered at the origin, with slight
    upward curl giving it a 3D shape.

    Returns:
        vertices: (N, 3)
        normals: (N, 3)
        colors: (N, 3) in [0, 1]
        faces: (M, 3) triangle indices
    """
    u = np.linspace(0, 1, n_radial)
    v = np.linspace(-1, 1, n_angular)
    U, V = np.meshgrid(u, v, indexing="ij")

    # Petal shape: width tapers to zero at base and tip
    width_envelope = np.sin(np.pi * U)
    x = U * petal_length
    y = V * petal_width * width_envelope
    z = curl_amount * U * U * (1.0 - 0.3 * V * V)

    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Compute normals via cross product of partial derivatives
    # du direction
    du_x = np.gradient(x, axis=0)
    du_y = np.gradient(y, axis=0)
    du_z = np.gradient(z, axis=0)
    # dv direction
    dv_x = np.gradient(x, axis=1)
    dv_y = np.gradient(y, axis=1)
    dv_z = np.gradient(z, axis=1)

    du = np.stack([du_x, du_y, du_z], axis=-1)
    dv = np.stack([dv_x, dv_y, dv_z], axis=-1)
    normals = np.cross(du, dv).reshape(-1, 3)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    # For degenerate vertices (base/tip of petal), use a default upward normal
    degenerate = (norms < 1e-6).squeeze()
    norms = np.maximum(norms, 1e-8)
    normals = normals / norms
    normals[degenerate] = np.array([0.0, 0.0, 1.0])

    # Colors: pink at base -> lighter pink at tip
    base_color = np.array([0.95, 0.3, 0.5])
    tip_color = np.array([1.0, 0.85, 0.9])
    t = U.reshape(-1, 1)
    colors = base_color * (1.0 - t) + tip_color * t

    # Generate triangle faces
    faces = []
    for i in range(n_radial - 1):
        for j in range(n_angular - 1):
            idx = i * n_angular + j
            faces.append([idx, idx + 1, idx + n_angular])
            faces.append([idx + 1, idx + n_angular + 1, idx + n_angular])
    faces = np.array(faces, dtype=np.int32)

    return vertices, normals, colors, faces


def generate_stem(
    n_segments: int = 10,
    n_around: int = 8,
    radius: float = 0.04,
    height: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a cylindrical stem along the -Y axis."""
    theta = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    y_vals = np.linspace(0, -height, n_segments)

    vertices = []
    normals = []
    for y in y_vals:
        for t in theta:
            vertices.append([radius * np.cos(t), y, radius * np.sin(t)])
            normals.append([np.cos(t), 0.0, np.sin(t)])

    vertices = np.array(vertices)
    normals = np.array(normals)
    colors = np.tile(np.array([0.2, 0.6, 0.15]), (len(vertices), 1))

    faces = []
    for i in range(n_segments - 1):
        for j in range(n_around):
            j_next = (j + 1) % n_around
            idx = i * n_around + j
            idx_next = i * n_around + j_next
            idx_below = idx + n_around
            idx_below_next = idx_next + n_around
            faces.append([idx, idx_next, idx_below])
            faces.append([idx_next, idx_below_next, idx_below])

    faces = np.array(faces, dtype=np.int32)
    return vertices, normals, colors, faces


def generate_pistil(
    n_lat: int = 8,
    n_lon: int = 12,
    radius: float = 0.12,
    center: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a spherical pistil (flower center)."""
    if center is None:
        center = np.array([0.0, 0.05, 0.0])

    phi = np.linspace(0, np.pi, n_lat)
    theta = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    PHI, THETA = np.meshgrid(phi, theta, indexing="ij")

    x = radius * np.sin(PHI) * np.cos(THETA) + center[0]
    y = radius * np.cos(PHI) + center[1]
    z = radius * np.sin(PHI) * np.sin(THETA) + center[2]

    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    normals_raw = vertices - center
    norms = np.linalg.norm(normals_raw, axis=-1, keepdims=True)
    normals = normals_raw / np.maximum(norms, 1e-8)

    colors = np.tile(np.array([0.95, 0.85, 0.2]), (len(vertices), 1))

    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            idx = i * n_lon + j
            idx_next = i * n_lon + j_next
            idx_below = idx + n_lon
            idx_below_next = idx_next + n_lon
            faces.append([idx, idx_next, idx_below])
            faces.append([idx_next, idx_below_next, idx_below])

    faces = np.array(faces, dtype=np.int32)
    return vertices, normals, colors, faces


def generate_flower(
    n_petals: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a complete flower: petals + stem + pistil.

    Returns:
        vertices: (N, 3) all vertex positions
        normals: (N, 3) per-vertex normals
        colors: (N, 3) per-vertex RGB in [0, 1]
        faces: (M, 3) triangle face indices
    """
    all_verts = []
    all_normals = []
    all_colors = []
    all_faces = []
    offset = 0

    # Generate petals with rotational symmetry
    for i in range(n_petals):
        angle = 2 * np.pi * i / n_petals
        v, n, c, f = generate_petal()

        # Rotation matrix around Y axis
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rot = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a],
        ])

        # Tilt the petal outward slightly
        tilt_angle = 0.3
        cos_t = np.cos(tilt_angle)
        sin_t = np.sin(tilt_angle)
        # Tilt around the local Z axis (after rotation)
        tilt = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1],
        ])

        transform = rot @ tilt
        v_transformed = (transform @ v.T).T
        n_transformed = (transform @ n.T).T

        all_verts.append(v_transformed)
        all_normals.append(n_transformed)
        all_colors.append(c)
        all_faces.append(f + offset)
        offset += len(v)

    # Stem
    v, n, c, f = generate_stem()
    all_verts.append(v)
    all_normals.append(n)
    all_colors.append(c)
    all_faces.append(f + offset)
    offset += len(v)

    # Pistil
    v, n, c, f = generate_pistil()
    all_verts.append(v)
    all_normals.append(n)
    all_colors.append(c)
    all_faces.append(f + offset)

    vertices = np.concatenate(all_verts, axis=0).astype(np.float32)
    normals = np.concatenate(all_normals, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors, axis=0).astype(np.float32)
    faces = np.concatenate(all_faces, axis=0).astype(np.int32)

    return vertices, normals, colors, faces
