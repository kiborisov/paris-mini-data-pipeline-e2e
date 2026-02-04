"""Stage 1: Download images from LAION-Aesthetic using img2dataset.

No GPU required. Downloads images, resizes to 256x256 center-crop,
and stores with metadata in parquet format.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

URL_KEYS = ("URL", "url")
CAPTION_KEYS = ("TEXT", "text", "caption", "description")
# Include common variants used across LAION indices.
AESTHETIC_KEYS = (
    "AESTHETIC_SCORE",
    "aesthetic_score",
    "AESTHETICS_SCORE",
    "aesthetics_score",
    "LAION_AESTHETIC_SCORE",
    "laion_aesthetic_score",
    "CLIP_SCORE",
    "clip_score",
    "AESTHETIC",
    "aesthetic",
)


def _get_value(sample: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    """Return the first non-null value for the provided keys."""
    for key in keys:
        value = sample.get(key)
        if value not in ("", None):
            return value
    return default


def _coerce_score(value: Any) -> float | None:
    """Best-effort convert to float. Returns None if missing/invalid."""
    if value in (None, ""):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score != score:  # NaN check
        return None
    return score


def _init_dataset(config: dict, skip_count: int = 0):
    """Create the LAION iterable dataset with retries + optional skip."""
    from datasets import DownloadConfig, load_dataset

    dataset_name = config.get("laion_index_dataset", "laion/laion2B-en-aesthetic")
    dataset_split = config.get("laion_index_split", "train")
    streaming = config.get("laion_index_streaming", True)

    download_config = DownloadConfig(
        max_retries=config.get("hf_stream_max_retries", 8),
        resume_download=True,
    )

    ds = load_dataset(
        dataset_name,
        split=dataset_split,
        streaming=streaming,
        download_config=download_config,
    )

    if hasattr(ds, "shuffle"):
        shuffle_kwargs = {"seed": config.get("random_seed", 42)}
        if streaming:
            shuffle_kwargs["buffer_size"] = min(
                10000,
                max(int(config["num_samples"] * config.get("ingest_oversample_factor", 1.0)), 1000),
            )
        ds = ds.shuffle(**shuffle_kwargs)

    if streaming and skip_count > 0 and hasattr(ds, "skip"):
        ds = ds.skip(skip_count)

    return ds


def _get_laion_subset_batch(config: dict, output_path: str):
    """Non-streaming: download a small slice of the dataset and filter locally.

    Downloads only the parquet shards needed for `laion_index_max_rows` rows,
    which is a reliable HTTP download instead of a fragile 2B-row stream.
    """
    from datasets import DownloadConfig, load_dataset

    dataset_name = config.get("laion_index_dataset", "laion/laion2B-en-aesthetic")
    dataset_split = config.get("laion_index_split", "train")
    max_rows = config.get("laion_index_max_rows", 200000)
    oversample = config.get("ingest_oversample_factor", 1.4)
    num_samples = config["num_samples"]
    min_aesthetic_score = config["min_aesthetic_score"]
    random_seed = config.get("random_seed", 42)

    target = max(int(num_samples * oversample), num_samples)
    split_str = f"{dataset_split}[:{max_rows}]"

    logger.info(
        "Downloading LAION-Aesthetic slice (%s, split=%s, %d rows)...",
        dataset_name,
        split_str,
        max_rows,
    )

    download_config = DownloadConfig(
        max_retries=config.get("hf_stream_max_retries", 8),
        resume_download=True,
    )

    ds = load_dataset(
        dataset_name,
        split=split_str,
        download_config=download_config,
    )

    logger.info("Converting to pandas and filtering (aesthetic >= %.1f)...", min_aesthetic_score)
    df_all = ds.to_pandas()

    logger.info("Dataset columns: %s", list(df_all.columns))

    # Normalize column names — LAION uses uppercase, find whichever exists
    url_col = next((c for c in df_all.columns if c.lower() in ("url",)), None)
    caption_col = next((c for c in df_all.columns if c.lower() in ("text", "caption", "description")), None)
    aesthetic_col = next(
        (
            c
            for c in df_all.columns
            if c.lower()
            in (
                "aesthetic_score",
                "aesthetics_score",
                "laion_aesthetic_score",
                "clip_score",
                "aesthetic",
            )
        ),
        None,
    )

    if url_col is None:
        raise RuntimeError(f"No URL column found in dataset. Columns: {list(df_all.columns)}")

    logger.info("Detected columns — url: %s, caption: %s, aesthetic: %s", url_col, caption_col, aesthetic_col)

    # Filter by aesthetic score
    if aesthetic_col:
        df_all[aesthetic_col] = pd.to_numeric(df_all[aesthetic_col], errors="coerce")
        df_filtered = df_all[df_all[aesthetic_col] >= min_aesthetic_score]
    else:
        # laion2B-en-aesthetic is already aesthetic-filtered, so all rows qualify
        logger.info("No aesthetic score column — dataset is pre-filtered, using all rows.")
        df_filtered = df_all

    logger.info("After aesthetic filter: %d / %d rows", len(df_filtered), len(df_all))

    # Sample down to target
    if len(df_filtered) > target:
        df_filtered = df_filtered.sample(target, random_state=random_seed)
    elif len(df_filtered) < num_samples:
        logger.warning(
            "Only %d rows pass aesthetic filter (need %d). Increase laion_index_max_rows.",
            len(df_filtered),
            num_samples,
        )

    # Build output with standardized column names
    # img2dataset reserves "caption" — so we store the text column as "caption"
    # which img2dataset auto-uses, and only list non-reserved cols in save_additional_columns
    if aesthetic_col:
        aesthetic_values = df_filtered[aesthetic_col].fillna(min_aesthetic_score).values
    else:
        aesthetic_values = min_aesthetic_score

    rows = pd.DataFrame({
        "url": df_filtered[url_col].values,
        "caption": df_filtered[caption_col].fillna("").values if caption_col else "",
        "aesthetic_score": aesthetic_values,
    })
    logger.info("Sample row: %s", rows.iloc[0].to_dict() if len(rows) > 0 else "empty")

    rows.to_parquet(output_path, index=False)
    logger.info("Saved %d URLs to %s", len(rows), output_path)
    return output_path


def _get_laion_subset_stream(config: dict, output_path: str):
    """Streaming fallback: iterate through the full dataset row by row.

    Used when laion_index_streaming is explicitly set to true.
    More fragile on unreliable connections but uses less disk space.
    """
    oversample = config.get("ingest_oversample_factor", 1.4)
    num_samples = config["num_samples"]
    min_aesthetic_score = config["min_aesthetic_score"]
    progress_interval = config.get("ingest_progress_interval", 1000)

    target = max(int(num_samples * oversample), num_samples)

    logger.info(
        "Streaming LAION-Aesthetic index (%s/%s, oversample=%sx target=%d)...",
        config.get("laion_index_dataset"),
        config.get("laion_index_split"),
        oversample,
        target,
    )

    rows = []
    ds = _init_dataset(config)
    iterator = iter(ds)
    stream_retry = 0
    stream_retry_limit = config.get("hf_stream_retry_limit", 5)
    stream_retry_base = config.get("hf_stream_retry_backoff", 5)

    while len(rows) < target:
        try:
            sample = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            stream_retry += 1
            if stream_retry > stream_retry_limit:
                raise RuntimeError(
                    f"Exceeded streaming retries ({stream_retry_limit}) while reading LAION index"
                ) from exc

            wait = min(stream_retry_base * (2 ** (stream_retry - 1)), 60)
            logger.warning(
                "Streaming error (%s). Retrying in %ds (attempt %d/%d)...",
                exc,
                wait,
                stream_retry,
                stream_retry_limit,
            )
            time.sleep(wait)
            ds = _init_dataset(config, skip_count=len(rows))
            iterator = iter(ds)
            continue

        stream_retry = 0
        url = _get_value(sample, URL_KEYS, "")
        if not url:
            continue

        raw_score = _get_value(sample, AESTHETIC_KEYS, None)
        score = _coerce_score(raw_score)
        if score is None:
            # Dataset is pre-filtered; keep the row but assign min score.
            score = min_aesthetic_score
        elif score < min_aesthetic_score:
            continue

        caption = _get_value(sample, CAPTION_KEYS, "")
        rows.append({
            "url": url,
            "caption": caption or "",
            "aesthetic_score": score,
        })

        if len(rows) % max(progress_interval, 1) == 0:
            logger.info("  Collected %d/%d URL candidates...", len(rows), target)

        if len(rows) >= target:
            break

    if len(rows) < num_samples:
        logger.warning(
            "Only collected %d URLs (requested %d). Stage 1 will continue but expect fewer downloads.",
            len(rows),
            num_samples,
        )

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} URLs to {output_path}")
    return output_path


def _get_laion_subset(config: dict, output_path: str):
    """Download a subset of LAION-Aesthetic URLs for processing.

    Uses batch download (non-streaming) by default for reliability.
    Falls back to streaming if explicitly configured.
    """
    streaming = config.get("laion_index_streaming", False)

    if streaming:
        return _get_laion_subset_stream(config, output_path)
    return _get_laion_subset_batch(config, output_path)


def run(config: dict) -> dict:
    """Download images from LAION-Aesthetic.

    Returns:
        Dict with stage metrics.
    """
    raw_dir = Path(config["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    num_samples = config["num_samples"]

    # Step 1: Get URL list
    url_parquet = str(raw_dir / "url_list.parquet")
    if not Path(url_parquet).exists():
        _get_laion_subset(config, url_parquet)
    else:
        logger.info(f"URL list already exists: {url_parquet}")

    # Step 2: Download images using img2dataset
    from img2dataset import download

    output_folder = str(raw_dir / "images")

    processes = config.get("img2dataset_processes", 4)
    threads = config.get("img2dataset_thread_count", 16)
    timeout = config.get("img2dataset_timeout", 10)
    retries = config.get("img2dataset_retries", 1)

    logger.info(
        "Downloading up to %d images to %s (processes=%d, threads=%d, timeout=%ss, retries=%d)...",
        num_samples,
        output_folder,
        processes,
        threads,
        timeout,
        retries,
    )

    download(
        url_list=url_parquet,
        output_folder=output_folder,
        image_size=256,
        resize_mode="center_crop",
        processes_count=processes,
        thread_count=threads,
        timeout=timeout,
        retries=retries,
        number_sample_per_shard=1000,
        output_format="files",
        save_additional_columns=["aesthetic_score"],
        input_format="parquet",
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

        raw_score = meta.get("aesthetic_score")
        score = _coerce_score(raw_score)
        if score is None:
            # laion2B-en-aesthetic is pre-filtered; default to min_aesthetic_score
            # so Stage 2 doesn't reject images that lack an explicit score
            score = config["min_aesthetic_score"]

        records.append({
            "image_path": str(img_path),
            "url": meta.get("url", ""),
            "original_caption": meta.get("caption", ""),
            "aesthetic_score": float(score),
        })

    if not records:
        raise RuntimeError("No images downloaded. Check internet access or dataset availability.")

    df = pd.DataFrame(records)
    if len(df) > num_samples:
        logger.info(
            "Trimming downloaded images from %d to target %d (random sample).",
            len(df),
            num_samples,
        )
        df = df.sample(num_samples, random_state=config.get("random_seed", 42)).reset_index(drop=True)

    metadata_path = str(raw_dir / "metadata.parquet")
    df.to_parquet(metadata_path, index=False)

    final_count = len(df)
    failed = max(num_samples - final_count, 0)

    logger.info(f"Download complete: {final_count} usable images, {failed} failed to download")

    return {
        "total_requested": num_samples,
        "total_downloaded": final_count,
        "failed_downloads": failed,
    }
