"""Inter-stage validation to catch silent data corruption early."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when inter-stage validation fails."""


def _check_parquet_exists(path: str, stage_name: str):
    if not Path(path).exists():
        raise ValidationError(f"[{stage_name}] Expected parquet not found: {path}")


def _check_no_nulls(df: pd.DataFrame, columns: list[str], stage_name: str):
    for col in columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                raise ValidationError(
                    f"[{stage_name}] Column '{col}' has {null_count} null values"
                )


def _check_min_rows(df: pd.DataFrame, min_rows: int, stage_name: str):
    if len(df) < min_rows:
        raise ValidationError(
            f"[{stage_name}] Expected >= {min_rows} rows, got {len(df)}"
        )


def _check_files_exist(df: pd.DataFrame, path_column: str, stage_name: str):
    if path_column not in df.columns:
        return
    missing = [p for p in df[path_column] if not Path(p).exists()]
    if missing:
        raise ValidationError(
            f"[{stage_name}] {len(missing)} referenced files do not exist. "
            f"First missing: {missing[0]}"
        )


def _check_embedding_shape(path: str, expected_dim: int, stage_name: str):
    if not Path(path).exists():
        raise ValidationError(f"[{stage_name}] Embedding file not found: {path}")
    emb = np.load(path)
    if emb.ndim != 2 or emb.shape[1] != expected_dim:
        raise ValidationError(
            f"[{stage_name}] Expected embedding shape (N, {expected_dim}), "
            f"got {emb.shape}"
        )


def validate_stage_output(stage_num: int, config: dict):
    """Run validation checks for the given stage's output.

    Raises ValidationError if checks fail.
    """
    validators = {
        1: _validate_ingest,
        2: _validate_filter,
        3: _validate_caption,
        4: _validate_cluster,
        5: _validate_validate_clusters,
        6: _validate_encode_latents,
        7: _validate_shard,
    }

    validator = validators.get(stage_num)
    if validator:
        validator(config)
        logger.info(f"Stage {stage_num} validation passed")


def _validate_ingest(config: dict):
    raw_dir = Path(config["raw_data_dir"])
    if not raw_dir.exists():
        raise ValidationError("[ingest] Raw data directory not found")

    parquet_path = raw_dir / "metadata.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        _check_min_rows(df, 1, "ingest")
        if "aesthetic_score" in df.columns:
            scores = pd.to_numeric(df["aesthetic_score"], errors="coerce")
            if scores.notnull().any():
                if (scores <= 0).all() and config.get("min_aesthetic_score", 0) > 0:
                    raise ValidationError(
                        "[ingest] aesthetic_score is 0 for all rows. "
                        "Check LAION column mapping in Stage 1."
                    )
        logger.info(f"[ingest] {len(df)} images downloaded")
    else:
        # img2dataset may use different output structure — check for any content
        files = list(raw_dir.rglob("*"))
        if not files:
            raise ValidationError("[ingest] Raw data directory is empty")
        logger.info(f"[ingest] {len(files)} files in raw data directory")


def _validate_filter(config: dict):
    parquet_path = Path(config["filtered_data_dir"]) / "metadata.parquet"
    _check_parquet_exists(parquet_path, "filter")

    df = pd.read_parquet(parquet_path)
    _check_min_rows(df, 1, "filter")
    _check_no_nulls(df, ["image_path"], "filter")
    _check_files_exist(df, "image_path", "filter")
    logger.info(f"[filter] {len(df)} images after filtering")


def _validate_caption(config: dict):
    parquet_path = Path(config["captioned_data_dir"]) / "metadata.parquet"
    _check_parquet_exists(parquet_path, "caption")

    df = pd.read_parquet(parquet_path)
    _check_no_nulls(df, ["image_path", "synthetic_caption"], "caption")

    clip_emb_path = Path(config["captioned_data_dir"]) / "clip_text_embeddings.npy"
    if clip_emb_path.exists():
        emb = np.load(clip_emb_path)
        if emb.shape[0] != len(df):
            raise ValidationError(
                f"[caption] CLIP embedding count ({emb.shape[0]}) != "
                f"metadata rows ({len(df)})"
            )
    logger.info(f"[caption] {len(df)} images captioned")


def _validate_cluster(config: dict):
    parquet_path = Path(config["clustered_data_dir"]) / "metadata.parquet"
    _check_parquet_exists(parquet_path, "cluster")

    df = pd.read_parquet(parquet_path)
    _check_no_nulls(df, ["image_path", "cluster_id"], "cluster")

    n_clusters = df["cluster_id"].nunique()
    expected = config["num_clusters"]
    if n_clusters != expected:
        raise ValidationError(
            f"[cluster] Expected {expected} clusters, found {n_clusters}"
        )

    emb_path = Path(config["clustered_data_dir"]) / "embeddings.npy"
    _check_embedding_shape(emb_path, 768, "cluster")

    # Check for severe imbalance
    threshold = config.get("cluster_imbalance_warn_threshold", 0.05)
    cluster_counts = df["cluster_id"].value_counts()
    for cid, count in cluster_counts.items():
        frac = count / len(df)
        if frac < threshold:
            logger.warning(
                f"[cluster] Cluster {cid} has only {frac:.1%} of data "
                f"({count} images) — below {threshold:.0%} threshold"
            )

    logger.info(f"[cluster] {len(df)} images across {n_clusters} clusters")


def _validate_validate_clusters(config: dict):
    quality_path = Path(config["analysis_dir"]) / "cluster_quality.json"
    if not quality_path.exists():
        raise ValidationError("[validate_clusters] cluster_quality.json not found")
    logger.info("[validate_clusters] Cluster quality report generated")


def _validate_encode_latents(config: dict):
    latent_dir = Path(config["latent_data_dir"])
    if not latent_dir.exists():
        raise ValidationError("[encode_latents] Latent directory not found")

    latent_files = list(latent_dir.glob("*.pt")) + list(latent_dir.glob("*.safetensors"))
    if not latent_files:
        raise ValidationError("[encode_latents] No latent files found")
    logger.info(f"[encode_latents] {len(latent_files)} latent files")


def _validate_shard(config: dict):
    shard_dir = Path(config["output_dir"]) / "expert_shards"
    if not shard_dir.exists():
        raise ValidationError("[shard] Expert shard directory not found")

    tar_files = list(shard_dir.rglob("*.tar"))
    if not tar_files:
        raise ValidationError("[shard] No .tar shard files found")

    manifest_path = Path(config["output_dir"]) / "pipeline_manifest.json"
    if not manifest_path.exists():
        raise ValidationError("[shard] pipeline_manifest.json not found")

    logger.info(f"[shard] {len(tar_files)} shard files across clusters")
