"""Stage 7: Package into WebDataset shards and router training data.

Produces two output sets:
1. Expert training shards: per-cluster .tar files containing VAE latents,
   CLIP text embeddings, captions, and metadata.
2. Router training data: (DINOv2 embedding, cluster_label) pairs for the
   full dataset.

No GPU required.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import webdataset as wds

from utils.helpers import load_metadata, sha256_file

logger = logging.getLogger(__name__)


def _build_expert_shards(df: pd.DataFrame, config: dict):
    """Create per-cluster WebDataset .tar shards."""
    output_dir = Path(config["output_dir"]) / "expert_shards"
    latent_dir = Path(config["latent_data_dir"])
    caption_dir = Path(config["captioned_data_dir"])
    shard_size = config["shard_size"]

    # Load CLIP embeddings
    clip_emb_path = caption_dir / "clip_text_embeddings.npy"
    clip_embeddings = np.load(str(clip_emb_path))

    n_clusters = config["num_clusters"]
    shard_checksums = {}

    for cluster_id in range(n_clusters):
        cluster_df = df[df["cluster_id"] == cluster_id].reset_index(drop=True)

        if len(cluster_df) == 0:
            logger.warning(f"Cluster {cluster_id} has no images, skipping")
            continue

        cluster_dir = output_dir / f"cluster_{cluster_id}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        pattern = str(cluster_dir / "shard-%04d.tar")

        with wds.ShardWriter(pattern, maxcount=shard_size) as sink:
            for _, row in cluster_df.iterrows():
                image_id = Path(row["image_path"]).stem

                # Load VAE latent
                latent_path = latent_dir / f"{image_id}.pt"
                if not latent_path.exists():
                    logger.warning(f"Missing latent for {image_id}, skipping")
                    continue

                latent = torch.load(latent_path, weights_only=True)

                # Get CLIP embedding by original index
                orig_idx = row.name if "orig_idx" in df.columns else None
                # We need to find the index in the captioned metadata
                clip_idx = _find_clip_index(row["image_path"], config)
                if clip_idx is not None and clip_idx < len(clip_embeddings):
                    clip_emb = clip_embeddings[clip_idx]
                else:
                    clip_emb = np.zeros((77, 768), dtype=np.float16)

                metadata = {
                    "cluster_id": int(row["cluster_id"]),
                    "aesthetic_score": float(row.get("aesthetic_score", 0)),
                    "cluster_distance": float(row.get("cluster_distance", 0)),
                    "original_url": row.get("url", ""),
                    "original_caption": row.get("original_caption", ""),
                }

                # Write to shard
                import io
                import pickle

                latent_buf = io.BytesIO()
                torch.save(latent, latent_buf)

                clip_buf = io.BytesIO()
                np.save(clip_buf, clip_emb)

                sink.write({
                    "__key__": image_id,
                    "latent.pth": latent_buf.getvalue(),
                    "clip_emb.npy": clip_buf.getvalue(),
                    "caption.txt": row.get("synthetic_caption", ""),
                    "metadata.json": json.dumps(metadata),
                })

        # Checksum shards
        for tar_file in sorted(cluster_dir.glob("*.tar")):
            rel_path = str(tar_file.relative_to(output_dir))
            shard_checksums[rel_path] = f"sha256:{sha256_file(str(tar_file))}"

        logger.info(
            f"Cluster {cluster_id}: {len(cluster_df)} images → "
            f"{len(list(cluster_dir.glob('*.tar')))} shards"
        )

    return shard_checksums


# Build a lookup index once for CLIP embedding mapping
_clip_index = None


def _find_clip_index(image_path: str, config: dict) -> int | None:
    """Find the CLIP embedding index for a given image path."""
    global _clip_index
    if _clip_index is None:
        caption_meta = load_metadata(
            str(Path(config["captioned_data_dir"]) / "metadata.parquet")
        )
        _clip_index = {
            path: idx for idx, path in enumerate(caption_meta["image_path"])
        }
    return _clip_index.get(image_path)


def _build_router_training_data(config: dict):
    """Package router training data: embeddings + cluster labels."""
    output_dir = Path(config["output_dir"]) / "router_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    clustered_dir = Path(config["clustered_data_dir"])

    # Copy embeddings and cluster labels
    embeddings = np.load(str(clustered_dir / "embeddings.npy"))
    df = load_metadata(str(clustered_dir / "metadata.parquet"))
    cluster_labels = df["cluster_id"].values.astype(np.int32)

    np.save(str(output_dir / "embeddings.npy"), embeddings)
    np.save(str(output_dir / "cluster_labels.npy"), cluster_labels)

    # Copy CLIP text embeddings if available
    clip_path = Path(config["captioned_data_dir"]) / "clip_text_embeddings.npy"
    if clip_path.exists():
        clip_emb = np.load(str(clip_path))
        np.save(str(output_dir / "clip_text_embeddings.npy"), clip_emb)

    logger.info(
        f"Router training data: {len(embeddings)} samples, "
        f"embeddings {embeddings.shape}, labels {cluster_labels.shape}"
    )


def _build_manifest(config: dict, shard_checksums: dict) -> dict:
    """Build the pipeline manifest with full provenance."""
    clustered_dir = Path(config["clustered_data_dir"])
    df = load_metadata(str(clustered_dir / "metadata.parquet"))

    unique, counts = np.unique(df["cluster_id"].values, return_counts=True)
    cluster_sizes = [int(counts[i]) for i in range(len(counts))]

    # Load cluster quality if available
    quality_path = Path(config["analysis_dir"]) / "cluster_quality.json"
    cluster_quality = {}
    if quality_path.exists():
        with open(quality_path) as f:
            quality = json.load(f)
        cluster_quality = {
            "silhouette_score": quality.get("silhouette_score"),
            "davies_bouldin_index": quality.get("davies_bouldin_index"),
        }

    # Compute content hash from shard checksums
    checksum_str = json.dumps(shard_checksums, sort_keys=True)
    content_hash = hashlib.sha256(checksum_str.encode()).hexdigest()[:8]

    manifest = {
        "pipeline_version": "1.0.0",
        "version_tag": f"v1.0.0-{content_hash}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": config["source"],
        "total_images_downloaded": config["num_samples"],
        "total_images_after_filter": len(df),
        "images_per_cluster": cluster_sizes,
        "models_used": {
            "captioning": config["caption_model"],
            "clip_text": config["clip_model"],
            "embedding": config["embedding_model"],
            "vae": config["vae_model"],
        },
        "cluster_quality": cluster_quality,
        "random_seeds": {
            "kmeans": config["random_seed"],
            "torch": config["random_seed"],
            "numpy": config["random_seed"],
        },
        "shard_format": "webdataset",
        "shard_contents": ["latent.pth", "clip_emb.npy", "caption.txt", "metadata.json"],
        "shard_checksums": shard_checksums,
    }

    manifest_path = Path(config["output_dir"]) / "pipeline_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest saved: {manifest_path} (version: {manifest['version_tag']})")

    return manifest


def run(config: dict) -> dict:
    """Package pipeline outputs into WebDataset shards and router training data.

    Returns:
        Dict with stage metrics.
    """
    # Merge clustered metadata with captions
    clustered_df = load_metadata(
        str(Path(config["clustered_data_dir"]) / "metadata.parquet")
    )
    captioned_df = load_metadata(
        str(Path(config["captioned_data_dir"]) / "metadata.parquet")
    )

    # Merge synthetic captions into clustered metadata
    if "synthetic_caption" in captioned_df.columns:
        caption_map = dict(
            zip(captioned_df["image_path"], captioned_df["synthetic_caption"])
        )
        clustered_df["synthetic_caption"] = clustered_df["image_path"].map(caption_map)
    else:
        clustered_df["synthetic_caption"] = ""

    # Build expert shards
    logger.info("Building expert training shards...")
    shard_checksums = _build_expert_shards(clustered_df, config)

    # Build router training data
    logger.info("Building router training data...")
    _build_router_training_data(config)

    # Build manifest
    logger.info("Building pipeline manifest...")
    manifest = _build_manifest(config, shard_checksums)

    total_shards = len(shard_checksums)

    return {
        "total_shards": total_shards,
        "version_tag": manifest["version_tag"],
        "images_per_cluster": manifest["images_per_cluster"],
    }
