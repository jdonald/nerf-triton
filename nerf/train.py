"""NeRF training loop."""

import json
import os

import numpy as np
import torch
from PIL import Image

from nerf.config import NeRFConfig
from nerf.model import NeRFModel
from nerf.rays import get_rays
from nerf.rendering import render_rays


def load_dataset(
    data_dir: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Load training images and camera poses.

    Args:
        data_dir: Directory containing transforms.json and images/
        device: torch device

    Returns:
        images: (N, H, W, 3) tensor of training images
        poses: (N, 4, 4) tensor of camera-to-world matrices
        focal: focal length in pixels
    """
    transforms_path = os.path.join(data_dir, "transforms.json")
    with open(transforms_path) as f:
        meta = json.load(f)

    focal = meta["focal"]
    frames = meta["frames"]

    images_list = []
    poses_list = []
    for frame in frames:
        img_path = os.path.join(data_dir, frame["file_path"])
        img = np.array(Image.open(img_path)).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]
        images_list.append(img)
        poses_list.append(np.array(frame["transform_matrix"]))

    images = torch.tensor(np.stack(images_list), dtype=torch.float32, device=device)
    poses = torch.tensor(np.stack(poses_list), dtype=torch.float32, device=device)
    return images, poses, focal


def train(
    config: NeRFConfig,
    data_dir: str = "data",
    output_dir: str = "checkpoints",
    device: torch.device = None,
    verbose: bool = True,
) -> tuple[NeRFModel, list[float]]:
    """Train a NeRF model.

    Args:
        config: Training configuration
        data_dir: Directory with training data
        output_dir: Directory for checkpoints
        device: torch device (auto-detected if None)
        verbose: Print progress

    Returns:
        Trained model and list of losses per epoch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    images, poses, focal = load_dataset(data_dir, device)
    n_images, H, W, _ = images.shape

    if verbose:
        print(f"Loaded {n_images} images of size {H}x{W}, focal={focal:.1f}")
        print(f"Device: {device}")

    # Create model
    model = NeRFModel(
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        skip_layer=config.skip_layer,
        num_freq_position=config.num_freq_position,
        num_freq_direction=config.num_freq_direction,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    losses = []
    step = 0
    for epoch in range(config.num_epochs):
        # Shuffle image order each epoch
        perm = torch.randperm(n_images)
        epoch_loss = 0.0
        n_batches = 0

        for img_idx in perm:
            img = images[img_idx]  # (H, W, 3)
            pose = poses[img_idx]  # (4, 4)

            # Generate all rays for this image
            rays_o, rays_d = get_rays(H, W, focal, pose)
            target_rgb = img.reshape(-1, 3)  # (H*W, 3)

            # Random subset of rays
            n_rays = rays_o.shape[0]
            if n_rays > config.batch_size:
                sel = torch.randperm(n_rays, device=device)[: config.batch_size]
                rays_o = rays_o[sel]
                rays_d = rays_d[sel]
                target_rgb = target_rgb[sel]

            # Render
            rgb_pred, _, _ = render_rays(
                model,
                rays_o,
                rays_d,
                near=config.near,
                far=config.far,
                num_samples=config.num_samples,
                perturb=True,
            )

            # MSE loss
            loss = torch.mean((rgb_pred - target_rgb) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

            # Learning rate decay
            if config.lr_decay_steps > 0 and step % config.lr_decay_steps == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= config.lr_decay_factor

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if verbose:
            print(f"Epoch {epoch + 1}/{config.num_epochs}  loss={avg_loss:.6f}")

    # Save checkpoint
    ckpt_path = os.path.join(output_dir, "model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "num_layers": config.num_layers,
                "hidden_dim": config.hidden_dim,
                "skip_layer": config.skip_layer,
                "num_freq_position": config.num_freq_position,
                "num_freq_direction": config.num_freq_direction,
            },
            "focal": focal,
            "image_height": H,
            "image_width": W,
            "near": config.near,
            "far": config.far,
            "num_samples": config.num_samples,
        },
        ckpt_path,
    )
    if verbose:
        print(f"Saved checkpoint to {ckpt_path}")

    return model, losses


def load_model(
    checkpoint_path: str,
    device: torch.device = None,
) -> tuple[NeRFModel, dict]:
    """Load a trained NeRF model from checkpoint.

    Returns:
        model: Loaded NeRF model
        meta: dict with focal, image_height, image_width, near, far, num_samples
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    model = NeRFModel(
        num_layers=cfg["num_layers"],
        hidden_dim=cfg["hidden_dim"],
        skip_layer=cfg["skip_layer"],
        num_freq_position=cfg["num_freq_position"],
        num_freq_direction=cfg["num_freq_direction"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {
        "focal": ckpt["focal"],
        "image_height": ckpt["image_height"],
        "image_width": ckpt["image_width"],
        "near": ckpt["near"],
        "far": ckpt["far"],
        "num_samples": ckpt["num_samples"],
    }
    return model, meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train NeRF")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--image_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    config = NeRFConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.lr,
    )
    train(config, args.data_dir, args.output_dir)
