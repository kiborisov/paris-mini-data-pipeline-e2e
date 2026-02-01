# Paris Data Pipeline — LAION to Expert Training Shards

A data pipeline that reproduces the data preparation strategy behind [Bagel Labs' Paris model](https://arxiv.org/abs/2510.03434) at demo scale.

Paris trains 8 independent DiT-XL/2 diffusion experts (605M params each) on semantically clustered data with zero inter-node communication. This pipeline produces everything needed to train both the experts and the post-hoc router: VAE latents, CLIP text embeddings, DINOv2-clustered partitions, and router supervision data.

## Pipeline

```
LAION-Aesthetic (10k images)
        │
   01_ingest ──── img2dataset download
        │
   02_filter ──── quality + NSFW + perceptual dedup (pHash)
        │
   03_caption ─── BLIP-2 captions + CLIP ViT-L/14 text embeddings
        │
   04_cluster ─── DINOv2-base embeddings → K-means (K=8)
        │
   05_validate ── silhouette score, visual grids, cluster report
        │
   06_encode ──── sd-vae-ft-mse → 32×32×4 latent tensors
        │
   07_shard ───── per-cluster WebDataset .tar + router training pairs
        │
   outputs/
   ├── expert_shards/cluster_{0-7}/*.tar
   ├── router_training/{embeddings,labels,clip}.npy
   └── pipeline_manifest.json
```

## Architecture Alignment

| Paris needs | This pipeline produces |
|---|---|
| 32×32×4 VAE latents per expert | Per-cluster WebDataset shards with pre-encoded latents |
| CLIP ViT-L/14 text conditioning | Pre-computed CLIP text embeddings in every shard |
| 8 semantic partitions (DINOv2) | K-means on DINOv2-base embeddings, validated with silhouette score |
| Router training data | (DINOv2 embedding, cluster_label) pairs for full dataset |

## Quick Start

```bash
pip install -r requirements.txt

# Full pipeline
python run_pipeline.py

# Resume from a specific stage (e.g., after session timeout)
python run_pipeline.py --resume-from 4

# Run specific stages only
python run_pipeline.py --stages 4,5

# Validate config without running
python run_pipeline.py --dry-run
```

**Kaggle (recommended)**: Create notebook → enable GPU T4 x2 → upload project → run.

## Results

### Cluster Quality

| Metric | Value |
|--------|-------|
| Silhouette score | *run pipeline to populate* |
| Davies-Bouldin index | *run pipeline to populate* |

### Cluster Distribution

| Cluster | Dominant Content | Sample Count |
|---------|-----------------|-------------|
| 0 | *run pipeline* | — |
| ... | ... | ... |
| 7 | *run pipeline* | — |

### Cluster Visualizations

*5×5 sample grids generated in `analysis/cluster_{0-7}_grid.png`*

### VAE Reconstruction Verification

*Side-by-side originals vs decoded latents in `analysis/vae_reconstruction_samples.png`*

## Scaling Notes

| Concern | 10k (demo) | 11M (Paris scale) |
|---------|-----------|-------------------|
| Clustering | sklearn K-means in-memory | faiss GPU K-means (billion-scale) |
| Captioning | Single T4, ~2 hrs | Multi-node vLLM, ~450 A40-hrs |
| Storage | ~6 GB local | ~1.9 TB S3, stream via WebDataset |
| Orchestration | `run_pipeline.py` | Prefect/Airflow DAG |

**What breaks first at 11M**: sklearn K-means — 11M × 768 float32 = 32 GB RAM. Switch to faiss GPU K-means. Captioning is the GPU bottleneck at ~900 GPU-hours on T4.

## Project Structure

```
├── config.yaml              # All pipeline parameters
├── run_pipeline.py          # Orchestrator with --resume-from
├── scripts/
│   ├── 01_ingest.py         # LAION-Aesthetic download
│   ├── 02_filter.py         # Quality filtering + dedup
│   ├── 03_caption.py        # BLIP-2 + CLIP text embeddings
│   ├── 04_cluster.py        # DINOv2 → K-means(8)
│   ├── 05_validate_clusters.py  # Cluster quality metrics + grids
│   ├── 06_encode_latents.py # VAE → 32×32×4 latents
│   └── 07_shard.py          # WebDataset shards + router data
├── utils/
│   ├── helpers.py           # Config, checkpoints, I/O
│   ├── logging_config.py    # Structured logging
│   └── validation.py        # Inter-stage validation
└── analysis/                # Generated cluster reports and grids
```

## References

- [Paris: A Decentralized Trained Open-Weight Diffusion Model](https://arxiv.org/abs/2510.03434) — Bagel Labs
- [LAION-Aesthetic](https://laion.ai/blog/laion-aesthetics/)
- [DINOv2](https://arxiv.org/abs/2304.07193) — Meta AI
- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) — Stability AI
- [CLIP ViT-L/14](https://github.com/openai/CLIP) — OpenAI
