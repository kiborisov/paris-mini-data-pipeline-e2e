"""Shared utilities: config loading, checkpointing, image I/O, seed management."""

import hashlib
import json
import logging
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Load pipeline configuration from YAML file."""
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def ensure_dirs(config: dict):
    """Create all output directories specified in config."""
    for key in config:
        if key.endswith("_dir"):
            Path(config[key]).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42):
    """Pin random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seeds set to {seed}")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(data, stage: str, checkpoint_dir: str = "checkpoints"):
    """Save intermediate results to a checkpoint file."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / f"{stage}_checkpoint.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.debug(f"Checkpoint saved: {path}")


def load_checkpoint(stage: str, checkpoint_dir: str = "checkpoints"):
    """Load checkpoint if it exists, otherwise return None."""
    path = Path(checkpoint_dir) / f"{stage}_checkpoint.pkl"
    if path.exists():
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Checkpoint loaded: {path}")
        return data
    return None


def clear_checkpoint(stage: str, checkpoint_dir: str = "checkpoints"):
    """Remove checkpoint file for a completed stage."""
    path = Path(checkpoint_dir) / f"{stage}_checkpoint.pkl"
    if path.exists():
        path.unlink()
        logger.debug(f"Checkpoint cleared: {path}")


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: str) -> Image.Image:
    """Load an image and convert to RGB."""
    return Image.open(path).convert("RGB")


def load_images_batch(paths: list[str]) -> list[Image.Image]:
    """Load a batch of images, skipping any that fail."""
    images = []
    for p in paths:
        try:
            images.append(load_image(p))
        except Exception as e:
            logger.warning(f"Failed to load {p}: {e}")
            images.append(None)
    return images


# ---------------------------------------------------------------------------
# Metadata I/O
# ---------------------------------------------------------------------------

def load_metadata(path: str) -> pd.DataFrame:
    """Load metadata parquet file."""
    return pd.read_parquet(path)


def save_metadata(df: pd.DataFrame, path: str):
    """Save metadata to parquet."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Metadata saved: {path} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require_gpu():
    """Raise an error if no GPU is available."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This stage requires a CUDA GPU. Run on Kaggle (T4 x2) or a machine with a GPU."
        )
