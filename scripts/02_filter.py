"""Stage 2: Quality filtering and perceptual deduplication.

Phase 1: Deterministic quality filters (resolution, aspect ratio, file integrity,
          aesthetic score, NSFW detection).
Phase 2: Perceptual deduplication using pHash.

GPU required for NSFW detection only.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from utils.helpers import get_device, load_metadata, save_metadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: Quality filters
# ---------------------------------------------------------------------------

def _check_resolution(img: Image.Image, min_res: int) -> bool:
    return img.width >= min_res and img.height >= min_res


def _check_aspect_ratio(img: Image.Image, max_ratio: float) -> bool:
    ratio = max(img.width, img.height) / max(min(img.width, img.height), 1)
    return ratio <= max_ratio


def _check_file_integrity(path: str) -> bool:
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False


def _run_nsfw_filter(image_paths: list[str], threshold: float) -> list[bool]:
    """Run NSFW detection on a batch of images. Returns list of bools (True = safe)."""
    from transformers import pipeline

    device = get_device()
    logger.info(f"Loading NSFW detector on {device}...")

    nsfw_pipe = pipeline(
        "image-classification",
        model="Falconsai/nsfw_image_detection",
        device=device,
    )

    results = []
    batch_size = 32

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (256, 256)))  # placeholder for failed loads

        preds = nsfw_pipe(images, batch_size=batch_size)

        for pred in preds:
            # pred is a list of {label, score} dicts
            nsfw_score = 0.0
            for item in pred:
                if item["label"].lower() in ("nsfw", "porn", "sexy", "hentai"):
                    nsfw_score = max(nsfw_score, item["score"])
            results.append(nsfw_score < threshold)

        if (i + batch_size) % 1000 == 0:
            logger.info(f"NSFW filter progress: {i + batch_size}/{len(image_paths)}")

    return results


def _apply_quality_filters(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Apply all quality filters. Returns filtered df and rejection stats."""
    min_res = config["min_resolution"]
    max_aspect = config["max_aspect_ratio"]
    min_aesthetic = config["min_aesthetic_score"]
    nsfw_threshold = config["nsfw_threshold"]

    rejection_reasons = Counter()
    keep_mask = np.ones(len(df), dtype=bool)

    logger.info("Running quality filters...")

    for idx, row in df.iterrows():
        path = row["image_path"]

        # File integrity
        if not _check_file_integrity(path):
            keep_mask[idx] = False
            rejection_reasons["file_integrity"] += 1
            continue

        try:
            img = Image.open(path)
        except Exception:
            keep_mask[idx] = False
            rejection_reasons["file_open_error"] += 1
            continue

        # Resolution
        if not _check_resolution(img, min_res):
            keep_mask[idx] = False
            rejection_reasons["resolution"] += 1
            continue

        # Aspect ratio
        if not _check_aspect_ratio(img, max_aspect):
            keep_mask[idx] = False
            rejection_reasons["aspect_ratio"] += 1
            continue

        # Aesthetic score
        if row.get("aesthetic_score", 0) < min_aesthetic:
            keep_mask[idx] = False
            rejection_reasons["aesthetic_score"] += 1
            continue

    # NSFW filter (batch, needs GPU)
    safe_indices = np.where(keep_mask)[0]
    safe_paths = df.iloc[safe_indices]["image_path"].tolist()

    if safe_paths:
        nsfw_safe = _run_nsfw_filter(safe_paths, nsfw_threshold)
        for i, is_safe in enumerate(nsfw_safe):
            if not is_safe:
                keep_mask[safe_indices[i]] = False
                rejection_reasons["nsfw"] += 1

    filtered_df = df[keep_mask].reset_index(drop=True)

    logger.info(f"Quality filtering: {len(df)} → {len(filtered_df)} images")
    for reason, count in rejection_reasons.most_common():
        logger.info(f"  Rejected ({reason}): {count}")

    return filtered_df, dict(rejection_reasons)


# ---------------------------------------------------------------------------
# Phase 2: Perceptual deduplication
# ---------------------------------------------------------------------------

def _deduplicate(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, int]:
    """Remove near-duplicate images using perceptual hashing.

    Buckets hashes by leading bits to avoid full O(n^2) comparisons.
    """
    hash_size = config.get("dedup_hash_size", 16)
    threshold = config.get("dedup_hamming_threshold", 6)
    bucket_bits = config.get("dedup_bucket_bits", 12)

    logger.info(
        "Computing perceptual hashes (hash_size=%d, bucket_bits=%d)...",
        hash_size,
        bucket_bits,
    )

    hashes = []
    hash_ints = []
    for _, row in df.iterrows():
        try:
            h = imagehash.phash(Image.open(row["image_path"]), hash_size=hash_size)
            hashes.append(h)
            hash_ints.append(int(str(h), 16))
        except Exception:
            hashes.append(None)
            hash_ints.append(None)

    hash_bits = hash_size * hash_size
    bucket_bits = min(bucket_bits, hash_bits)
    shift = max(hash_bits - bucket_bits, 0)

    buckets: dict[int, list[int]] = defaultdict(list)
    for idx, h_int in enumerate(hash_ints):
        if h_int is None:
            continue
        bucket_key = h_int >> shift if shift else h_int
        buckets[bucket_key].append(idx)

    df = df.copy()
    df["_phash"] = hashes
    df["_is_dup"] = False

    aesthetic_list = df["aesthetic_score"].tolist()
    is_dup = [False] * len(df)

    for bucket_idx, indices in buckets.items():
        if len(indices) <= 1:
            continue
        for pos_i in range(len(indices)):
            i = indices[pos_i]
            if is_dup[i]:
                continue
            hash_i = df["_phash"].iat[i]
            if hash_i is None:
                continue
            for pos_j in range(pos_i + 1, len(indices)):
                j = indices[pos_j]
                if is_dup[j]:
                    continue
                hash_j = df["_phash"].iat[j]
                if hash_j is None:
                    continue
                if hash_i - hash_j <= threshold:
                    if aesthetic_list[i] >= aesthetic_list[j]:
                        is_dup[j] = True
                    else:
                        is_dup[i] = True
                        break

    n_dups = sum(is_dup)
    df_deduped = df[~pd.Series(is_dup)].drop(columns=["_phash", "_is_dup"]).reset_index(drop=True)

    logger.info(
        "Deduplication: removed %d near-duplicates, %d remaining (buckets=%d)",
        n_dups,
        len(df_deduped),
        len(buckets),
    )

    return df_deduped, n_dups


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config: dict) -> dict:
    """Run filtering and deduplication.

    Returns:
        Dict with stage metrics.
    """
    raw_metadata = Path(config["raw_data_dir"]) / "metadata.parquet"
    df = load_metadata(str(raw_metadata))
    initial_count = len(df)

    # Phase 1: Quality filters
    df, rejection_stats = _apply_quality_filters(df, config)

    # Phase 2: Deduplication
    df, n_dups = _deduplicate(df, config)

    # Save
    output_path = Path(config["filtered_data_dir"]) / "metadata.parquet"
    save_metadata(df, str(output_path))

    return {
        "initial_count": initial_count,
        "after_filter": len(df) + n_dups,
        "duplicates_removed": n_dups,
        "final_count": len(df),
        "rejection_stats": rejection_stats,
    }
