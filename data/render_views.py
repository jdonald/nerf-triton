"""Render training views of the flower scene using a software rasterizer.

This avoids any OpenGL dependency for data generation, making it portable
across environments (headless servers, CI, etc.).
"""

import json
import os

import numpy as np
from PIL import Image

from data.generate_flower import generate_flower
from nerf.rays import look_at


def rasterize_triangle(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    n0: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    image: np.ndarray,
    zbuf: np.ndarray,
    light_dir: np.ndarray,
    ambient: float = 0.3,
):
    """Rasterize a single triangle with z-buffer and Lambertian shading."""
    H, W = zbuf.shape

    # Bounding box
    xs = [v0[0], v1[0], v2[0]]
    ys = [v0[1], v1[1], v2[1]]
    min_x = max(int(np.floor(min(xs))), 0)
    max_x = min(int(np.ceil(max(xs))), W - 1)
    min_y = max(int(np.floor(min(ys))), 0)
    max_y = min(int(np.ceil(max(ys))), H - 1)

    if min_x > max_x or min_y > max_y:
        return

    # Edge vectors for barycentric coordinates
    e01 = v1[:2] - v0[:2]
    e02 = v2[:2] - v0[:2]
    det = e01[0] * e02[1] - e01[1] * e02[0]
    if abs(det) < 1e-10:
        return
    inv_det = 1.0 / det

    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            p = np.array([px + 0.5, py + 0.5])
            ep = p - v0[:2]
            u = (ep[0] * e02[1] - ep[1] * e02[0]) * inv_det
            v = (e01[0] * ep[1] - e01[1] * ep[0]) * inv_det
            w = 1.0 - u - v

            if u >= 0 and v >= 0 and w >= 0:
                z = w * v0[2] + u * v1[2] + v * v2[2]
                if z < zbuf[py, px]:
                    zbuf[py, px] = z

                    # Interpolate color and normal
                    color = w * c0 + u * c1 + v * c2
                    normal = w * n0 + u * n1 + v * n2
                    norm_len = np.linalg.norm(normal)
                    if norm_len > 1e-8:
                        normal = normal / norm_len

                    # Lambertian shading
                    diffuse = max(0.0, np.dot(normal, light_dir))
                    shade = ambient + (1.0 - ambient) * diffuse
                    image[py, px] = np.clip(color * shade, 0, 1)


def project_vertices(
    vertices: np.ndarray,
    c2w: np.ndarray,
    focal: float,
    height: int,
    width: int,
) -> np.ndarray:
    """Project 3D vertices to 2D screen coordinates.

    Args:
        vertices: (N, 3) world-space positions
        c2w: (4, 4) camera-to-world matrix
        focal: focal length in pixels
        height: image height
        width: image width

    Returns:
        (N, 3) screen-space positions [x, y, depth]
    """
    # World-to-camera
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    # Transform to camera space
    cam_pts = (R @ vertices.T).T + t  # (N, 3)

    # OpenGL: camera looks down -Z, so depth = -z
    depth = -cam_pts[:, 2]

    # Project to screen
    screen = np.zeros((len(vertices), 3))
    valid = depth > 0.01
    screen[valid, 0] = focal * cam_pts[valid, 0] / (-cam_pts[valid, 2]) + width * 0.5
    screen[valid, 1] = -focal * cam_pts[valid, 1] / (-cam_pts[valid, 2]) + height * 0.5
    screen[valid, 2] = depth[valid]
    screen[~valid, 2] = 1e10  # push behind camera points far away

    return screen


def transform_normals(normals: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """Transform normals to camera space (for lighting in camera space)."""
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    cam_normals = (R @ normals.T).T
    norms = np.linalg.norm(cam_normals, axis=-1, keepdims=True)
    return cam_normals / np.maximum(norms, 1e-8)


def render_scene(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    c2w: np.ndarray,
    focal: float,
    height: int,
    width: int,
    background: np.ndarray = None,
) -> np.ndarray:
    """Render the scene from a given camera pose.

    Args:
        vertices: (N, 3) positions
        faces: (M, 3) triangle indices
        colors: (N, 3) per-vertex RGB
        normals: (N, 3) per-vertex normals
        c2w: (4, 4) camera-to-world matrix
        focal: focal length in pixels
        height: image height
        width: image width
        background: (3,) background color, default white

    Returns:
        (H, W, 3) rendered image in [0, 1]
    """
    if background is None:
        background = np.array([1.0, 1.0, 1.0])

    # Project vertices
    screen = project_vertices(vertices, c2w, focal, height, width)

    # Light direction in camera space (from upper-right-front)
    light_world = np.array([0.5, 0.8, 0.3])
    light_world = light_world / np.linalg.norm(light_world)
    # Keep lighting in world space for consistency
    light_dir = light_world

    image = np.full((height, width, 3), background, dtype=np.float64)
    zbuf = np.full((height, width), np.inf, dtype=np.float64)

    # Sort faces by depth (painter's algorithm backup — zbuffer is primary)
    face_depths = np.mean(screen[faces, 2], axis=1)
    sorted_indices = np.argsort(-face_depths)  # back to front

    for fi in sorted_indices:
        f = faces[fi]
        v0, v1, v2 = screen[f[0]], screen[f[1]], screen[f[2]]

        # Skip if any vertex is behind camera
        if v0[2] <= 0 or v1[2] <= 0 or v2[2] <= 0:
            continue

        rasterize_triangle(
            v0, v1, v2,
            colors[f[0]], colors[f[1]], colors[f[2]],
            normals[f[0]], normals[f[1]], normals[f[2]],
            image, zbuf, light_dir,
        )

    return image.astype(np.float32)


def generate_camera_poses(
    n_views: int,
    radius: float = 3.0,
    target: np.ndarray = None,
) -> list[np.ndarray]:
    """Generate camera poses on a hemisphere looking at the target.

    Cameras are distributed on the upper hemisphere using a Fibonacci
    spiral for approximately uniform spacing.
    """
    if target is None:
        target = np.array([0.0, -0.2, 0.0])

    poses = []
    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in range(n_views):
        # Fibonacci sphere (upper hemisphere only)
        theta = 2 * np.pi * i / golden_ratio
        # phi from 0 (top) to pi/2 (equator), but with more coverage near equator
        phi = np.arccos(1 - (i + 0.5) / n_views)
        phi = min(phi, np.pi * 0.45)  # limit to upper hemisphere

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.cos(phi)
        z = radius * np.sin(phi) * np.sin(theta)

        cam_pos = np.array([x, y, z])
        c2w = look_at(cam_pos, target)
        poses.append(c2w)

    return poses


def generate_dataset(
    output_dir: str = "data",
    n_views: int = 100,
    image_size: int = 100,
    radius: float = 3.0,
) -> dict:
    """Generate the complete training dataset.

    Args:
        output_dir: Directory to save images and transforms
        n_views: Number of training views
        image_size: Image height and width
        radius: Camera distance from origin

    Returns:
        dict with 'camera_angle_x', 'focal', and 'frames' list
    """
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Generate flower mesh
    vertices, normals, colors, faces = generate_flower()

    # Camera intrinsics
    fov_x = 0.6911  # ~39.6 degrees
    focal = image_size * 0.5 / np.tan(fov_x * 0.5)

    # Generate poses
    poses = generate_camera_poses(n_views, radius=radius)

    frames = []
    all_images = []
    for i, c2w in enumerate(poses):
        img = render_scene(
            vertices, faces, colors, normals,
            c2w, focal, image_size, image_size,
        )
        all_images.append(img)

        # Save image
        img_path = os.path.join(images_dir, f"r_{i:03d}.png")
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(img_path)

        frames.append({
            "file_path": f"./images/r_{i:03d}.png",
            "transform_matrix": c2w.tolist(),
        })

    # Save transforms
    transforms = {
        "camera_angle_x": fov_x,
        "focal": focal,
        "frames": frames,
    }
    transforms_path = os.path.join(output_dir, "transforms.json")
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"Generated {n_views} views at {image_size}x{image_size} in {output_dir}/")
    return transforms


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate NeRF training data")
    parser.add_argument("--output_dir", default="data", help="Output directory")
    parser.add_argument("--n_views", type=int, default=100, help="Number of views")
    parser.add_argument("--image_size", type=int, default=100, help="Image size")
    parser.add_argument("--radius", type=float, default=3.0, help="Camera radius")
    args = parser.parse_args()

    generate_dataset(args.output_dir, args.n_views, args.image_size, args.radius)
