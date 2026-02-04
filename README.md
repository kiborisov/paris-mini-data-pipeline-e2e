# Paris Data Pipeline — LAION to Expert Training Shards

A data pipeline that reproduces the data preparation strategy behind [Bagel Labs' Paris model](https://arxiv.org/abs/2510.03434) at demo scale.

Paris trains 8 independent DiT-XL/2 diffusion experts (605M params each) on semantically clustered data with zero inter-node communication. This pipeline produces everything needed to train both the experts and the post-hoc router: VAE latents, CLIP text embeddings, DINOv2-clustered partitions, and router supervision data.

## Intro

I came across the Paris paper (arXiv 2510.03434) from Bagel Labs while reading about decentralized training approaches. The core idea is unique. You take 8 independent DiT-XL/2 diffusion experts at 605M parameters each, train them on semantically clustered partitions of the data with zero inter-node communication, then route between them at inference time using a lightweight DiT-B router around 129M params.

What caught my attention was the data side. Paris clusters its training set using DINOv2 embeddings and conditions generation on CLIP ViT-L/14 text embeddings, with images encoded to 32x32x4 latents through Stability AI's sd-vae-ft-mse VAE. The whole thing falls apart if the clustering is bad or the preprocessing is sloppy. Eight experts are only as good as the eight partitions they train on. I wanted to understand what that data pipeline actually looks like end to end, how you go from raw LAION images to clean, clustered, VAE-encoded shards that are ready for distributed training.

So I built one. Not at Paris scale (11M images from LAION-Aesthetic needs a few hundred GPU-hours and proper orchestration), but at 10k images on a Kaggle T4 x2 instance, enough to validate every stage and hit real edge cases. The pipeline has 7 stages: ingest from laion/laion2B-en-aesthetic on Hugging Face, quality filtering with NSFW detection and pHash perceptual deduplication, captioning with BLIP-2 (Salesforce/blip2-opt-2.7b) plus CLIP ViT-L/14 text embedding extraction, DINOv2-base visual embedding followed by K-means clustering into 8 partitions, cluster validation with silhouette score and Davies-Bouldin index, VAE latent encoding to 32x32x4 float16 tensors, and finally packaging everything into per-cluster WebDataset tar shards with router training pairs.

Out of 10,000 requested images, 9,096 came back (904 dead URLs), and 8,994 survived filtering (53 flagged NSFW, 49 perceptual duplicates removed). The 8 clusters landed on recognizable semantic groupings: cars and vehicles (645 samples), dresses and fashion (544), paintings and fine art (544), illustrations and clipart (548), food and beverages (1,143), general photography as a catch-all (3,746), crafts and decorative objects (1,370), and cakes and confections (454). Silhouette score came out at 0.0425 and Davies-Bouldin at 5.0757, which is expected at this scale with the catch-all cluster absorbing 41.6% of the data. The tighter clusters showed intra-cluster cosine similarity between 0.17 and 0.30, which indicates meaningful semantic separation. The final output is 17 WebDataset shards totaling about 4.28 GB of latents, CLIP embeddings, BLIP-2 captions, and metadata, plus numpy arrays for router training. The whole run took about 38 minutes wall time.

It was not a clean run on the first try. Streaming directly from LAION kept dropping connections, so I switched to downloading 200k rows upfront and sampling from there, which made the ingest stage reliable. The aesthetic score column was missing from the initial metadata pull, which meant every shard came out with aesthetic 0.00 across the board. I fixed that by pulling the column through properly, re-ran, and then noticed 4 out of 17 total shard files still had zeros on aesthetic scores. Those turned out to be stale tar files from the first run that the sharding script (07_shard.py) did not clean up before writing new ones. There is also a possibility that some of the original LAION records simply did not have aesthetic scores attached to them. As a next step I would either compute aesthetic scores myself using something like LAION's aesthetic predictor or re-run the pipeline with stricter validation on the metadata to confirm where the gaps are.

I also built a shard viewer tool (FastAPI + React) to browse the pipeline outputs interactively. It reads the WebDataset tar files directly and gives you a cluster overview dashboard with total images, shard counts, and size distribution. It runs health checks to spot missing, extra, or zero-size shard files, which is how I caught the 4 stale shards that were not in the manifest. You can filter entries by aesthetic score, cluster distance, or whether they have a URL or caption, and paginate through results. Images are streamed directly from the raw files with a fallback to reading from the shards. There is also an entry detail modal with a larger preview, captions, scores, and source URL. It started as a debugging tool but ended up being the fastest way to audit data quality without writing throwaway scripts.

![Shard viewer overview](assets/shard_viewer_overview.png)
![Shard viewer entries](assets/shard_viewer_entries.png)

The whole point was to get hands on with the data engineering behind mixture-of-experts diffusion training and see what breaks at each stage. It was a great learning experience, and the pipeline is here if anyone wants to reproduce it or build on top of it. Once I fix the aesthetic score gaps and validate the data is clean across all shards, I plan to publish the dataset on Hugging Face for others to use.

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

> Stage 1 streams `laion/laion2B-en-aesthetic` by default (matching Bagel's Paris pipeline). If Hugging Face ever republishes a smaller curated subset, you can switch `laion_index_*` in `config.yaml`, but for now the full LAION stream is the supported path.

## Results

Pipeline run on 10,000 LAION-Aesthetic images (Kaggle T4 x2). 8,994 images survived filtering (53 NSFW, 49 duplicates). Total wall time: ~38 minutes.

### Pipeline Metrics

| Stage | Detail | Time |
|-------|--------|------|
| Ingest | 9,096 / 10,000 downloaded (904 dead URLs) | 7.2 min |
| Filter + Dedup | 8,994 retained (53 NSFW, 49 pHash dupes) | 2.8 min |
| Caption + CLIP | BLIP-2 captions + CLIP ViT-L/14 embeddings | 20.5 min |
| Cluster | DINOv2-base → K-means(8) | 3.0 min |
| Validate | Silhouette, D-B, grids | 0.2 min |
| Encode | sd-vae-ft-mse → 32×32×4 fp16 latents (70 MB) | 4.4 min |
| Shard | 13 WebDataset .tar shards | 0.3 min |

### Cluster Quality

| Metric | Value |
|--------|-------|
| Silhouette score | 0.0425 |
| Davies-Bouldin index | 5.0757 |

The low silhouette score is expected at 10k scale with only 8 clusters over a diverse aesthetic dataset. Cluster 5 acts as a catch-all for general photography (41.6% of data), pulling the global score down. The tighter clusters (0, 1, 2, 3, 7) show intra-cluster cosine similarity 0.17–0.30, indicating meaningful semantic grouping.

### Cluster Distribution

| Cluster | Dominant Content | Samples | Fraction |
|---------|-----------------|---------|----------|
| 0 | Cars & vehicles | 645 | 7.2% |
| 1 | Dresses & fashion | 544 | 6.0% |
| 2 | Paintings & fine art | 544 | 6.0% |
| 3 | Illustrations & clipart | 548 | 6.1% |
| 4 | Food & beverages | 1,143 | 12.7% |
| 5 | General photography (catch-all) | 3,746 | 41.6% |
| 6 | Crafts & decorative objects | 1,370 | 15.2% |
| 7 | Cakes & confections | 454 | 5.1% |

### Cluster Visualizations

5x5 sample grids per cluster:

<details>
<summary>Click to expand cluster grids</summary>

| | |
|---|---|
| ![Cluster 0](assets/cluster_0_grid.png) | ![Cluster 1](assets/cluster_1_grid.png) |
| ![Cluster 2](assets/cluster_2_grid.png) | ![Cluster 3](assets/cluster_3_grid.png) |
| ![Cluster 4](assets/cluster_4_grid.png) | ![Cluster 5](assets/cluster_5_grid.png) |
| ![Cluster 6](assets/cluster_6_grid.png) | ![Cluster 7](assets/cluster_7_grid.png) |

</details>

### VAE Reconstruction Verification

Side-by-side originals vs decoded latents:

![VAE reconstruction samples](assets/vae_reconstruction_samples.png)

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
