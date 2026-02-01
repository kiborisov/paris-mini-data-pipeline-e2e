"""Stage 5: Quantitative and visual cluster validation.

Produces the evidence that the pipeline works — silhouette scores,
Davies-Bouldin index, per-cluster coherence metrics, and visual sample grids.

No GPU required.
"""

import json
import logging
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from utils.helpers import load_metadata, set_seeds

logger = logging.getLogger(__name__)


def _compute_cluster_metrics(
    embeddings: np.ndarray,
    cluster_ids: np.ndarray,
    centroids: np.ndarray,
) -> dict:
    """Compute quantitative cluster quality metrics."""
    embeddings_norm = normalize(embeddings)

    # Global metrics
    logger.info("Computing silhouette score (this may take a moment)...")
    sample_size = min(5000, len(embeddings_norm))
    sil_score = silhouette_score(
        embeddings_norm,
        cluster_ids,
        metric="cosine",
        sample_size=sample_size,
        random_state=42,
    )

    db_score = davies_bouldin_score(embeddings_norm, cluster_ids)

    logger.info(f"Silhouette score: {sil_score:.4f}")
    logger.info(f"Davies-Bouldin index: {db_score:.4f}")

    # Per-cluster metrics
    cluster_metrics = {}
    n_clusters = len(np.unique(cluster_ids))

    for cid in range(n_clusters):
        mask = cluster_ids == cid
        cluster_emb = embeddings_norm[mask]
        n_samples = int(mask.sum())

        # Intra-cluster cosine similarity (sample if large)
        if n_samples > 500:
            sample_idx = np.random.choice(n_samples, 500, replace=False)
            sim_matrix = cosine_similarity(cluster_emb[sample_idx])
        else:
            sim_matrix = cosine_similarity(cluster_emb)

        # Exclude self-similarity (diagonal)
        np.fill_diagonal(sim_matrix, 0)
        intra_sim = sim_matrix.sum() / (sim_matrix.shape[0] * (sim_matrix.shape[0] - 1))

        # Boundary fraction: images whose second-nearest centroid is < 1.2x
        # distance to nearest centroid
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(cluster_emb, normalize(centroids))
        sorted_dists = np.sort(dists, axis=1)
        if sorted_dists.shape[1] >= 2:
            ratios = sorted_dists[:, 1] / np.maximum(sorted_dists[:, 0], 1e-8)
            boundary_frac = float(np.mean(ratios < 1.2))
        else:
            boundary_frac = 0.0

        cluster_metrics[int(cid)] = {
            "n_samples": n_samples,
            "fraction": round(n_samples / len(cluster_ids), 4),
            "intra_cluster_cosine_sim": round(float(intra_sim), 4),
            "boundary_fraction": round(boundary_frac, 4),
        }

        logger.info(
            f"  Cluster {cid}: n={n_samples}, "
            f"intra_sim={intra_sim:.3f}, "
            f"boundary={boundary_frac:.1%}"
        )

    return {
        "silhouette_score": round(float(sil_score), 4),
        "davies_bouldin_index": round(float(db_score), 4),
        "per_cluster": cluster_metrics,
    }


def _plot_cluster_grids(
    df,
    cluster_ids: np.ndarray,
    output_dir: str,
    grid_size: int = 5,
):
    """Generate a 5x5 sample grid for each cluster."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_clusters = len(np.unique(cluster_ids))

    for cid in range(n_clusters):
        mask = cluster_ids == cid
        cluster_paths = df.loc[mask, "image_path"].tolist()
        n_samples = min(grid_size * grid_size, len(cluster_paths))

        if n_samples == 0:
            continue

        sampled = random.sample(cluster_paths, n_samples)

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle(
            f"Cluster {cid} (n={len(cluster_paths)})",
            fontsize=16,
            fontweight="bold",
        )

        for ax_idx, ax in enumerate(axes.flat):
            if ax_idx < len(sampled):
                try:
                    img = Image.open(sampled[ax_idx]).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Error", ha="center", va="center")
            ax.axis("off")

        grid_path = output_path / f"cluster_{cid}_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Grid saved: {grid_path}")


def _plot_cluster_distribution(cluster_ids: np.ndarray, output_dir: str):
    """Generate a bar chart of cluster sizes."""
    unique, counts = np.unique(cluster_ids, return_counts=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(unique, counts, color="#4a90d9", edgecolor="white")
    ax.set_xlabel("Cluster ID", fontsize=12)
    ax.set_ylabel("Image Count", fontsize=12)
    ax.set_title("Cluster Size Distribution", fontsize=14, fontweight="bold")
    ax.set_xticks(unique)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    path = Path(output_dir) / "cluster_size_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Distribution chart saved: {path}")


def _generate_report(metrics: dict, output_dir: str):
    """Generate a markdown report with cluster analysis."""
    report_lines = [
        "# Cluster Validation Report\n",
        "## Global Metrics\n",
        f"- **Silhouette Score**: {metrics['silhouette_score']:.4f}",
        f"- **Davies-Bouldin Index**: {metrics['davies_bouldin_index']:.4f}\n",
        "## Per-Cluster Analysis\n",
        "| Cluster | Samples | Fraction | Intra-Sim | Boundary % |",
        "|---------|---------|----------|-----------|------------|",
    ]

    for cid, cm in sorted(metrics["per_cluster"].items()):
        report_lines.append(
            f"| {cid} | {cm['n_samples']} | {cm['fraction']:.1%} | "
            f"{cm['intra_cluster_cosine_sim']:.3f} | {cm['boundary_fraction']:.1%} |"
        )

    report_lines.extend([
        "\n## Cluster Visualizations\n",
    ])
    for cid in sorted(metrics["per_cluster"].keys()):
        report_lines.append(f"![Cluster {cid}](cluster_{cid}_grid.png)\n")

    report_lines.append("\n![Cluster Distribution](cluster_size_distribution.png)\n")

    report_path = Path(output_dir) / "cluster_report.md"
    report_path.write_text("\n".join(report_lines))
    logger.info(f"Report saved: {report_path}")


def run(config: dict) -> dict:
    """Run cluster validation and generate analysis artifacts.

    Returns:
        Dict with stage metrics.
    """
    set_seeds(config["random_seed"])

    clustered_dir = Path(config["clustered_data_dir"])
    analysis_dir = config["analysis_dir"]

    df = load_metadata(str(clustered_dir / "metadata.parquet"))
    embeddings = np.load(str(clustered_dir / "embeddings.npy"))
    centroids = np.load(str(clustered_dir / "kmeans_centroids.npy"))
    cluster_ids = df["cluster_id"].values

    # Compute metrics
    logger.info("Computing cluster quality metrics...")
    metrics = _compute_cluster_metrics(embeddings, cluster_ids, centroids)

    # Save metrics JSON
    Path(analysis_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(analysis_dir) / "cluster_quality.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Visual grids
    logger.info("Generating cluster sample grids...")
    _plot_cluster_grids(df, cluster_ids, analysis_dir)

    # Distribution chart
    _plot_cluster_distribution(cluster_ids, analysis_dir)

    # Markdown report
    _generate_report(metrics, analysis_dir)

    return {
        "silhouette_score": metrics["silhouette_score"],
        "davies_bouldin_index": metrics["davies_bouldin_index"],
        "num_clusters": len(metrics["per_cluster"]),
    }
