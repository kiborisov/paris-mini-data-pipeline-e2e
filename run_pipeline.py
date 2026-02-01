#!/usr/bin/env python3
"""Pipeline orchestrator — single entry point for all stages.

Usage:
    python run_pipeline.py                        # Run full pipeline
    python run_pipeline.py --resume-from 4        # Resume from stage 4 (cluster)
    python run_pipeline.py --stages 4,5           # Run specific stages only
    python run_pipeline.py --dry-run              # Validate config without executing
    python run_pipeline.py --config custom.yaml   # Use custom config file
"""

import argparse
import importlib
import json
import logging
import sys
import time
from pathlib import Path

from utils.helpers import ensure_dirs, load_config, set_seeds
from utils.logging_config import reset_stage_timer, setup_logging
from utils.validation import ValidationError, validate_stage_output

logger = logging.getLogger(__name__)

STAGES = {
    1: ("ingest", "scripts.01_ingest"),
    2: ("filter_dedup", "scripts.02_filter"),
    3: ("caption_clip", "scripts.03_caption"),
    4: ("cluster", "scripts.04_cluster"),
    5: ("validate_clusters", "scripts.05_validate_clusters"),
    6: ("encode_latents", "scripts.06_encode_latents"),
    7: ("shard", "scripts.07_shard"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paris data pipeline — LAION to expert training shards"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=1,
        help="Stage number to resume from (default: 1, runs all)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated list of specific stages to run (e.g., '4,5')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print execution plan without running",
    )
    return parser.parse_args()


def validate_config(config: dict):
    """Check that all required config keys are present."""
    required_keys = [
        "source",
        "num_samples",
        "min_aesthetic_score",
        "min_resolution",
        "caption_model",
        "clip_model",
        "embedding_model",
        "vae_model",
        "num_clusters",
        "shard_size",
        "raw_data_dir",
        "filtered_data_dir",
        "captioned_data_dir",
        "clustered_data_dir",
        "latent_data_dir",
        "output_dir",
        "checkpoint_dir",
        "analysis_dir",
        "random_seed",
    ]
    missing = [k for k in required_keys if k not in config]
    if missing:
        logger.error(f"Missing config keys: {missing}")
        sys.exit(1)
    logger.info("Config validation passed")


def main():
    args = parse_args()
    config = load_config(args.config)

    setup_logging(config.get("log_level", "INFO"))
    validate_config(config)

    # Determine which stages to run
    if args.stages:
        stage_nums = [int(s.strip()) for s in args.stages.split(",")]
    else:
        stage_nums = [n for n in STAGES if n >= args.resume_from]

    # Print execution plan
    logger.info("=" * 60)
    logger.info("Paris Data Pipeline")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Stages to run: {stage_nums}")
    logger.info(f"Source: {config['source']} ({config['num_samples']} samples)")
    logger.info(f"Clusters: {config['num_clusters']}")
    logger.info(f"Random seed: {config['random_seed']}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN — validating config and stage dependencies only")
        for num in stage_nums:
            stage_name, module_path = STAGES[num]
            logger.info(f"  Stage {num}: {stage_name} ({module_path})")
        logger.info("Dry run complete. Config is valid.")
        return

    # Create directories
    ensure_dirs(config)
    set_seeds(config["random_seed"])

    # Run stages
    all_metrics = {}
    pipeline_start = time.time()

    for stage_num in stage_nums:
        stage_name, module_path = STAGES[stage_num]

        logger.info("")
        logger.info(f"{'=' * 60}")
        logger.info(f"Stage {stage_num}: {stage_name}")
        logger.info(f"{'=' * 60}")

        reset_stage_timer()
        t0 = time.time()

        try:
            module = importlib.import_module(module_path)
            stage_metrics = module.run(config)
        except Exception as e:
            logger.error(f"Stage {stage_num} ({stage_name}) failed: {e}")
            logger.error("Pipeline halted. Fix the issue and resume with:")
            logger.error(f"  python run_pipeline.py --resume-from {stage_num}")
            sys.exit(1)

        elapsed = time.time() - t0
        stage_metrics["elapsed_seconds"] = round(elapsed, 1)
        all_metrics[stage_name] = stage_metrics

        logger.info(f"Stage {stage_num} complete in {elapsed:.1f}s")

        # Inter-stage validation
        try:
            validate_stage_output(stage_num, config)
        except ValidationError as e:
            logger.error(f"Validation failed after stage {stage_num}: {e}")
            logger.error("Pipeline halted. Inspect outputs and re-run stage.")
            sys.exit(1)

    total_elapsed = time.time() - pipeline_start

    # Save pipeline metrics
    all_metrics["total_elapsed_seconds"] = round(total_elapsed, 1)
    metrics_path = Path(config["output_dir"]) / "pipeline_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {total_elapsed:.1f}s")
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
