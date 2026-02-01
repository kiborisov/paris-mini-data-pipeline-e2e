"""Stage 1: Download images from LAION-Aesthetic using img2dataset.

No GPU required. Downloads images, resizes to 256x256 center-crop,
and stores with metadata in parquet format.
"""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _get_laion_subset(num_samples: int, min_aesthetic_score: float, output_path: str):
    """Download a subset of LAION-Aesthetic URLs for processing.

    Uses the HuggingFace datasets library to fetch the parquet index,
    then samples the requested number of rows.
    """
    from datasets import load_dataset

    logger.info(
        f"Loading LAION-Aesthetic index (requesting {num_samples} samples, "
        f"min aesthetic score {min_aesthetic_score})..."
    )

    ds = load_dataset(
        "laion/laion2B-en-aesthetic",
        split="train",
        streaming=True,
    )

    rows = []
    for sample in ds:
        if sample.get("AESTHETIC_SCORE", 0) >= min_aesthetic_score:
            rows.append({
                "url": sample["URL"],
                "caption": sample.get("TEXT", ""),
                "aesthetic_score": sample.get("AESTHETIC_SCORE", 0),
            })
        if len(rows) >= num_samples:
            break

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} URLs to {output_path}")
    return output_path


def run(config: dict) -> dict:
    """Download images from LAION-Aesthetic.

    Returns:
        Dict with stage metrics.
    """
    raw_dir = Path(config["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    num_samples = config["num_samples"]
    min_aesthetic = config["min_aesthetic_score"]

    # Step 1: Get URL list
    url_parquet = str(raw_dir / "url_list.parquet")
    if not Path(url_parquet).exists():
        _get_laion_subset(num_samples, min_aesthetic, url_parquet)
    else:
        logger.info(f"URL list already exists: {url_parquet}")

    # Step 2: Download images using img2dataset
    from img2dataset import download

    output_folder = str(raw_dir / "images")

    logger.info(f"Downloading {num_samples} images to {output_folder}...")

    download(
        url_list=url_parquet,
        output_folder=output_folder,
        image_size=256,
        resize_mode="center_crop",
        thread_count=16,
        number_sample_per_shard=1000,
        output_format="files",
        save_additional_columns=["aesthetic_score", "caption"],
        enable_wandb=False,
    )

    # Step 3: Build metadata parquet from downloaded images
    image_dir = Path(output_folder)
    image_files = sorted(image_dir.rglob("*.jpg")) + sorted(image_dir.rglob("*.png"))
    image_files += sorted(image_dir.rglob("*.webp"))

    records = []
    for img_path in image_files:
        json_path = img_path.with_suffix(".json")
        meta = {}
        if json_path.exists():
            import json
            with open(json_path) as f:
                meta = json.load(f)

        records.append({
            "image_path": str(img_path),
            "url": meta.get("url", ""),
            "original_caption": meta.get("caption", ""),
            "aesthetic_score": float(meta.get("aesthetic_score", 0)),
        })

    df = pd.DataFrame(records)
    metadata_path = str(raw_dir / "metadata.parquet")
    df.to_parquet(metadata_path, index=False)

    total_downloaded = len(df)
    failed = num_samples - total_downloaded

    logger.info(f"Download complete: {total_downloaded} images, {failed} failed")

    return {
        "total_requested": num_samples,
        "total_downloaded": total_downloaded,
        "failed_downloads": failed,
    }
