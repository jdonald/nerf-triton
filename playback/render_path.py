"""Render a recorded camera path through a trained NeRF model.

Produces a sequence of PNG images that can be played back at 60fps
or assembled into a video via ffmpeg.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from nerf.rendering import render_image
from nerf.train import load_model
from viewer.recorder import PathRecorder


def render_camera_path(
    checkpoint_path: str,
    path_file: str,
    output_dir: str = "output/frames",
    render_height: int = None,
    render_width: int = None,
    device: torch.device = None,
    verbose: bool = True,
) -> list[str]:
    """Render each frame from a recorded camera path.

    Args:
        checkpoint_path: Path to trained model checkpoint
        path_file: Path to camera_path.json
        output_dir: Directory for output frame images
        render_height: Override image height (default: use training size)
        render_width: Override image width (default: use training size)
        device: Torch device
        verbose: Print progress

    Returns:
        List of output image file paths
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, meta = load_model(checkpoint_path, device)

    H = render_height or meta["image_height"]
    W = render_width or meta["image_width"]
    focal = meta["focal"]
    near = meta["near"]
    far = meta["far"]
    num_samples = meta["num_samples"]

    # Scale focal length if rendering at different resolution
    if render_height and render_height != meta["image_height"]:
        focal = focal * render_height / meta["image_height"]

    # Load camera path
    frames = PathRecorder.load(path_file)

    if verbose:
        print(f"Rendering {len(frames)} frames at {H}x{W}")
        print(f"Model: {checkpoint_path}")
        print(f"Device: {device}")

    output_paths = []
    for i, frame in enumerate(frames):
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=device)

        with torch.no_grad():
            rgb = render_image(
                model, c2w, H, W, focal,
                near=near, far=far,
                num_samples=num_samples,
                chunk_size=4096,
            )

        # Convert to uint8 image
        rgb_np = rgb.cpu().numpy()
        rgb_uint8 = (np.clip(rgb_np, 0, 1) * 255).astype(np.uint8)

        frame_path = os.path.join(output_dir, f"frame_{i:05d}.png")
        Image.fromarray(rgb_uint8).save(frame_path)
        output_paths.append(frame_path)

        if verbose:
            print(f"  Frame {i + 1}/{len(frames)}: {frame_path}")

    if verbose:
        print(f"Done. {len(output_paths)} frames saved to {output_dir}/")
        print(f"To make a video: ffmpeg -framerate 60 -i {output_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4")

    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render NeRF from camera path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--path", required=True, help="Camera path JSON file")
    parser.add_argument("--output", default="output/frames", help="Output directory")
    parser.add_argument("--height", type=int, default=None, help="Render height")
    parser.add_argument("--width", type=int, default=None, help="Render width")
    args = parser.parse_args()

    render_camera_path(
        args.checkpoint, args.path, args.output,
        render_height=args.height, render_width=args.width,
    )
