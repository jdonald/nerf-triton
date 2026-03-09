"""Capture the four sample screenshots used in the README.

Produces:
    1. opengl.png   – scene rendered via the software rasterization pipeline
                      (same triangle / z-buffer / Lambertian-shading algorithm
                       used by the interactive OpenGL viewer)
    2. nerf_front.png       – NeRF novel view, front angle
    3. nerf_side.png        – NeRF novel view, side-elevated angle
    4. nerf_back_tele.png   – NeRF novel view, back angle with tighter FOV

Usage:
    # Generate training data first (skipped automatically if already present):
    python -m data.render_views --output_dir data/screenshots --n_views 30 --image_size 64

    # Then run this script:
    python -m scripts.capture_screenshots

    # Or do everything in one step (slower first run):
    python -m scripts.capture_screenshots --regen_data --retrain
"""

import argparse
import math
import os

import numpy as np
import torch
from PIL import Image

from data.generate_flower import generate_flower
from data.render_views import generate_dataset, render_scene
from nerf.config import NeRFConfig
from nerf.rays import look_at
from nerf.rendering import render_image
from nerf.train import load_model, train

# ---------------------------------------------------------------------------
# Camera specifications
# ---------------------------------------------------------------------------

# The four screenshot camera definitions.
# pos: camera world position
# target: world point the camera looks at
# focal_scale: multiplier applied to the training focal length (1.0 = same FOV)
SCREENSHOT_CAMERAS = [
    # 1 ── conventional rasterisation pipeline (same angle as default viewer)
    {
        "name": "opengl",
        "pos": np.array([0.0, 0.5, 3.0]),
        "target": np.array([0.0, -0.2, 0.0]),
        "focal_scale": 1.0,
        "kind": "rasterized",
    },
    # 2 ── NeRF, front view (direct comparison with OpenGL frame)
    {
        "name": "nerf_front",
        "pos": np.array([0.0, 0.5, 3.0]),
        "target": np.array([0.0, -0.2, 0.0]),
        "focal_scale": 1.0,
        "kind": "nerf",
    },
    # 3 ── NeRF, side/elevated (120° around, +20° elevation)
    {
        "name": "nerf_side",
        "pos": np.array([2.5, 1.5, -1.5]),
        "target": np.array([0.0, -0.2, 0.0]),
        "focal_scale": 1.0,
        "kind": "nerf",
    },
    # 4 ── NeRF, back with narrower FOV (telephoto-like parameterisation)
    {
        "name": "nerf_back_tele",
        "pos": np.array([-1.8, 0.6, -2.5]),
        "target": np.array([0.0, -0.2, 0.0]),
        "focal_scale": 1.5,   # longer focal → tighter / telephoto FOV
        "kind": "nerf",
    },
]

# ---------------------------------------------------------------------------
# Training config for screenshot generation
# ---------------------------------------------------------------------------

SCREENSHOT_CONFIG = NeRFConfig(
    # Full-capacity model
    num_layers=8,
    hidden_dim=256,
    skip_layer=4,
    num_freq_position=10,
    num_freq_direction=4,
    # Sampling
    num_samples=64,
    near=2.0,
    far=6.0,
    # Training — 100 views gives 100 grad steps/epoch.
    # lr_decay kicks in at step 3500 (epoch 35); gentle 0.5 factor keeps
    # learning alive through epoch 40.  This reaches loss ~0.009 in ~30 min.
    learning_rate=5e-4,
    num_epochs=35,
    batch_size=1024,
    lr_decay_steps=3500,
    lr_decay_factor=0.5,
    # Data
    image_size=100,
    num_views=100,
    # Rendering
    render_image_size=200,
    chunk_size=4096,
    backend="pytorch",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_rasterized(cam: dict, render_size: int) -> np.ndarray:
    """Render a single frame with the software rasteriser."""
    vertices, normals, colors, faces = generate_flower()
    fov_x = 0.6911
    focal = render_size * 0.5 / np.tan(fov_x * 0.5)
    c2w = look_at(cam["pos"], cam["target"])
    img = render_scene(vertices, faces, colors, normals, c2w, focal, render_size, render_size)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _render_nerf(cam: dict, model, meta: dict, render_size: int,
                 device: torch.device) -> np.ndarray:
    """Render a single NeRF frame."""
    focal = meta["focal"] * cam["focal_scale"] * render_size / meta["image_height"]
    c2w = look_at(cam["pos"], cam["target"])
    c2w_t = torch.tensor(c2w, dtype=torch.float32, device=device)
    with torch.no_grad():
        rgb = render_image(
            model, c2w_t,
            render_size, render_size, focal,
            near=meta["near"], far=meta["far"],
            num_samples=meta["num_samples"],
            chunk_size=4096,
        )
    rgb_np = rgb.cpu().numpy()
    return (np.clip(rgb_np, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def capture_screenshots(
    output_dir: str = "docs/screenshots",
    data_dir: str = "data/screenshots_data",
    checkpoint_dir: str = "checkpoints/screenshots",
    render_size: int = 200,
    regen_data: bool = False,
    retrain: bool = False,
) -> list[str]:
    """Run the full pipeline and save the four screenshot images.

    Args:
        output_dir: Where to write the PNG files.
        data_dir: Training data directory (generated if absent or regen_data=True).
        checkpoint_dir: Checkpoint directory (trained if absent or retrain=True).
        render_size: Pixel width/height of the output images (square).
        regen_data: Force regeneration of training data even if it exists.
        retrain: Force retraining even if a checkpoint exists.

    Returns:
        List of paths to the saved screenshot files.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Generate training data ------------------------------------------------
    transforms_path = os.path.join(data_dir, "transforms.json")
    if regen_data or not os.path.exists(transforms_path):
        print("Generating training data …")
        generate_dataset(
            output_dir=data_dir,
            n_views=SCREENSHOT_CONFIG.num_views,
            image_size=SCREENSHOT_CONFIG.image_size,
        )
    else:
        print(f"Using existing training data in {data_dir}/")

    # 2. Train (or load) the NeRF model ----------------------------------------
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    if retrain or not os.path.exists(checkpoint_path):
        print("Training NeRF …")
        train(SCREENSHOT_CONFIG, data_dir=data_dir, output_dir=checkpoint_dir, device=device)
    else:
        print(f"Loading existing checkpoint from {checkpoint_path}")

    model, meta = load_model(checkpoint_path, device)
    model.eval()

    # 3. Render all four screenshots --------------------------------------------
    saved_paths = []
    for cam in SCREENSHOT_CAMERAS:
        out_path = os.path.join(output_dir, f"{cam['name']}.png")
        print(f"Rendering {cam['name']} ({cam['kind']}) …")

        if cam["kind"] == "rasterized":
            img = _render_rasterized(cam, render_size)
        else:
            img = _render_nerf(cam, model, meta, render_size, device)

        Image.fromarray(img).save(out_path)
        saved_paths.append(out_path)
        print(f"  Saved → {out_path}")

    print(f"\nAll {len(saved_paths)} screenshots saved to {output_dir}/")
    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture README sample screenshots (rasterized + NeRF)"
    )
    parser.add_argument("--output_dir", default="docs/screenshots",
                        help="Directory for output PNGs (default: docs/screenshots)")
    parser.add_argument("--data_dir", default="data/screenshots_data",
                        help="Training data directory")
    parser.add_argument("--checkpoint_dir", default="checkpoints/screenshots",
                        help="Model checkpoint directory")
    parser.add_argument("--render_size", type=int, default=200,
                        help="Output image size in pixels (default: 200)")
    parser.add_argument("--regen_data", action="store_true",
                        help="Regenerate training data even if it exists")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain the model even if a checkpoint exists")
    args = parser.parse_args()

    capture_screenshots(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        render_size=args.render_size,
        regen_data=args.regen_data,
        retrain=args.retrain,
    )
