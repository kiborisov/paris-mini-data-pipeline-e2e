"""Stage 4: DINOv2 embedding extraction and K-means clustering.

Partitions images into 8 semantically coherent groups matching Paris's
8 expert architecture. Uses DINOv2-base CLS token embeddings with
L2-normalization before K-means.

GPU required.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm

from utils.helpers import (
    get_device,
    load_checkpoint,
    load_image,
    load_metadata,
    require_gpu,
    save_checkpoint,
    save_metadata,
    set_seeds,
)

logger = logging.getLogger(__name__)


def _extract_embeddings(image_paths: list[str], config: dict) -> np.ndarray:
    """Extract DINOv2 CLS token embeddings for all images.

    Returns: (N, 768) float32 array.
    """
    from transformers import AutoImageProcessor, AutoModel

    device = get_device()
    model_name = config["embedding_model"]
    batch_size = config["embedding_batch_size"]

    logger.info(f"Loading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Check for checkpoint
    checkpoint = load_checkpoint("embed", config["checkpoint_dir"])
    start_idx = 0
    all_embeddings = []

    if checkpoint:
        all_embeddings = checkpoint["embeddings"]
        start_idx = checkpoint["next_idx"]
        logger.info(f"Resuming embedding from index {start_idx}")

    for i in tqdm(range(start_idx, len(image_paths), batch_size), desc="DINOv2 embedding"):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            try:
                images.append(load_image(p))
            except Exception:
                # Use a blank image as placeholder for failed loads
                images.append(load_image(batch_paths[0]))

        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0]  # CLS token
            all_embeddings.append(cls_embeddings.cpu().numpy())

        if (i + batch_size) % 1000 < batch_size:
            save_checkpoint(
                {"embeddings": all_embeddings, "next_idx": i + batch_size},
                "embed",
                config["checkpoint_dir"],
            )

    del model, processor
    torch.cuda.empty_cache()

    return np.vstack(all_embeddings)


def _cluster_embeddings(embeddings: np.ndarray, config: dict) -> tuple[np.ndarray, KMeans]:
    """Run K-means clustering on L2-normalized embeddings.

    Returns: (cluster_ids array, fitted KMeans object).
    """
    n_clusters = config["num_clusters"]
    seed = config["random_seed"]

    logger.info(f"L2-normalizing {embeddings.shape[0]} embeddings...")
    embeddings_norm = normalize(embeddings)

    logger.info(
        f"Running K-means: K={n_clusters}, n_init={config['kmeans_n_init']}, "
        f"max_iter={config['kmeans_max_iter']}"
    )
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=config["kmeans_n_init"],
        max_iter=config["kmeans_max_iter"],
        random_state=seed,
    )
    cluster_ids = kmeans.fit_predict(embeddings_norm)

    # Log cluster distribution
    unique, counts = np.unique(cluster_ids, return_counts=True)
    for cid, count in zip(unique, counts):
        frac = count / len(cluster_ids)
        logger.info(f"  Cluster {cid}: {count} images ({frac:.1%})")

    return cluster_ids, kmeans


def run(config: dict) -> dict:
    """Run DINOv2 embedding extraction and K-means clustering.

    Returns:
        Dict with stage metrics.
    """
    require_gpu()
    set_seeds(config["random_seed"])

    df = load_metadata(str(Path(config["filtered_data_dir"]) / "metadata.parquet"))
    image_paths = df["image_path"].tolist()

    # Step 1: Extract embeddings
    logger.info(f"Extracting DINOv2 embeddings for {len(image_paths)} images...")
    embeddings = _extract_embeddings(image_paths, config)

    # Step 2: Cluster
    cluster_ids, kmeans = _cluster_embeddings(embeddings, config)

    # Compute distances to assigned centroid
    embeddings_norm = normalize(embeddings)
    all_distances = kmeans.transform(embeddings_norm)
    assigned_distances = all_distances[np.arange(len(cluster_ids)), cluster_ids]

    # Update metadata
    df["cluster_id"] = cluster_ids
    df["cluster_distance"] = assigned_distances

    # Save outputs
    output_dir = Path(config["clustered_data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    save_metadata(df, str(output_dir / "metadata.parquet"))

    emb_path = str(output_dir / "embeddings.npy")
    np.save(emb_path, embeddings.astype(np.float16))
    logger.info(f"Embeddings saved: {embeddings.shape} to {emb_path}")

    centroid_path = str(output_dir / "kmeans_centroids.npy")
    np.save(centroid_path, kmeans.cluster_centers_)
    logger.info(f"Centroids saved: {kmeans.cluster_centers_.shape} to {centroid_path}")

    # Cluster summary
    import json
    unique, counts = np.unique(cluster_ids, return_counts=True)
    summary = {
        "num_clusters": int(config["num_clusters"]),
        "total_images": len(cluster_ids),
        "cluster_sizes": {int(k): int(v) for k, v in zip(unique, counts)},
        "inertia": float(kmeans.inertia_),
    }
    summary_path = str(output_dir / "cluster_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Clear checkpoint
    from utils.helpers import clear_checkpoint
    clear_checkpoint("embed", config["checkpoint_dir"])

    return {
        "total_images": len(cluster_ids),
        "num_clusters": int(config["num_clusters"]),
        "cluster_sizes": {int(k): int(v) for k, v in zip(unique, counts)},
        "kmeans_inertia": float(kmeans.inertia_),
    }
