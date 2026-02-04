"""Stage 6: VAE latent encoding.

Encodes images to 32x32x4 latent tensors using sd-vae-ft-mse,
matching the input format Paris experts train on. Pre-encoding avoids
redundant VAE forward passes during every training epoch.

GPU required.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from utils.helpers import (
    get_device,
    load_checkpoint,
    load_image,
    load_metadata,
    require_gpu,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

# VAE preprocessing: scale to [-1, 1] as expected by sd-vae-ft-mse
VAE_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def _encode_images(image_paths: list[str], config: dict, latent_dir: Path) -> int:
    """Encode all images to VAE latent space and stream latents to disk."""
    from diffusers import AutoencoderKL

    device = get_device()
    vae_model = config["vae_model"]
    batch_size = config["vae_batch_size"]
    scaling_factor = config["vae_scaling_factor"]

    logger.info(f"Loading VAE: {vae_model}...")
    vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=torch.float16).to(device)
    vae.eval()

    checkpoint = load_checkpoint("vae_encode", config["checkpoint_dir"])
    start_idx = checkpoint.get("next_idx", 0) if checkpoint else 0
    if start_idx > 0:
        logger.info(f"Resuming VAE encoding from index {start_idx}")

    total_encoded = 0

    for i in tqdm(range(start_idx, len(image_paths), batch_size), desc="VAE encoding"):
        batch_paths = image_paths[i:i + batch_size]
        pending = []
        tensors = []

        for p in batch_paths:
            image_id = Path(p).stem
            latent_path = latent_dir / f"{image_id}.pt"
            if latent_path.exists():
                continue
            try:
                img = load_image(p)
                tensors.append(VAE_PREPROCESS(img))
            except Exception:
                tensors.append(torch.zeros(3, 256, 256))
            pending.append((image_id, latent_path))

        if not pending:
            save_checkpoint({"next_idx": i + batch_size}, "vae_encode", config["checkpoint_dir"])
            continue

        batch = torch.stack(tensors).to(device, dtype=torch.float16)

        with torch.no_grad():
            posterior = vae.encode(batch).latent_dist
            latents = posterior.sample() * scaling_factor

        for latent_tensor, (image_id, latent_path) in zip(latents, pending):
            torch.save(latent_tensor.cpu(), latent_path)
            total_encoded += 1

        if (i + batch_size) % 1000 < batch_size:
            save_checkpoint({"next_idx": i + batch_size}, "vae_encode", config["checkpoint_dir"])

    del vae
    torch.cuda.empty_cache()

    return total_encoded


def _verify_reconstructions(latent_dir: Path, image_paths: list[str], config: dict):
    """Decode a few random latents and save reconstruction comparison."""
    from diffusers import AutoencoderKL

    device = get_device()
    vae = AutoencoderKL.from_pretrained(
        config["vae_model"], torch_dtype=torch.float16
    ).to(device)
    vae.eval()

    import random
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [p for p in image_paths if (latent_dir / f"{Path(p).stem}.pt").exists()]
    if not available:
        logger.warning("No latents available for reconstruction verification.")
        return

    sample_paths = random.sample(available, min(10, len(available)))
    fig, axes = plt.subplots(2, len(sample_paths), figsize=(3 * len(sample_paths), 6))

    for idx, path in enumerate(sample_paths):
        image_id = Path(path).stem
        latent_path = latent_dir / f"{image_id}.pt"
        try:
            latent = torch.load(latent_path, map_location=device, weights_only=True)
        except TypeError:
            latent = torch.load(latent_path, map_location=device)
        latent = latent.unsqueeze(0).to(device, dtype=torch.float16)

        with torch.no_grad():
            decoded = vae.decode(latent / config["vae_scaling_factor"]).sample

        orig = load_image(path)
        axes[0, idx].imshow(orig)
        axes[0, idx].set_title("Original", fontsize=8)
        axes[0, idx].axis("off")

        recon = decoded[0].cpu().float()
        recon = (recon * 0.5 + 0.5).clamp(0, 1)
        recon = recon.permute(1, 2, 0).numpy()
        axes[1, idx].imshow(recon)
        axes[1, idx].set_title("Reconstructed", fontsize=8)
        axes[1, idx].axis("off")

    plt.suptitle("VAE Reconstruction Verification", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = Path(config["analysis_dir"]) / "vae_reconstruction_samples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Reconstruction verification saved: {output_path}")

    del vae
    torch.cuda.empty_cache()


def run(config: dict) -> dict:
    """Run VAE latent encoding for all filtered images.

    Returns:
        Dict with stage metrics.
    """
    require_gpu()

    df = load_metadata(str(Path(config["filtered_data_dir"]) / "metadata.parquet"))
    image_paths = df["image_path"].tolist()

    latent_dir = Path(config["latent_data_dir"])
    latent_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Encoding {len(image_paths)} images to VAE latent space...")
    newly_encoded = _encode_images(image_paths, config, latent_dir)

    latent_files = list(latent_dir.glob("*.pt"))
    if not latent_files:
        raise RuntimeError("No latent tensors found after encoding stage.")

    logger.info(
        "Latent encoding complete: %d new, %d total stored.",
        newly_encoded,
        len(latent_files),
    )

    # Verify reconstructions
    logger.info("Generating reconstruction verification...")
    _verify_reconstructions(latent_dir, image_paths, config)

    # Clear checkpoint
    from utils.helpers import clear_checkpoint
    clear_checkpoint("vae_encode", config["checkpoint_dir"])

    # Stats
    try:
        sample_latent = torch.load(latent_files[0], map_location="cpu", weights_only=True)
    except TypeError:
        sample_latent = torch.load(latent_files[0], map_location="cpu")
    total_bytes = len(latent_files) * sample_latent.nelement() * sample_latent.element_size()

    return {
        "images_encoded": len(latent_files),
        "latent_shape": list(sample_latent.shape),
        "total_storage_mb": round(total_bytes / (1024 * 1024), 2),
        "dtype": str(sample_latent.dtype),
    }
