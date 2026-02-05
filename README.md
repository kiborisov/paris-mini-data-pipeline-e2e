# Paris Data Pipeline - LAION to Expert Training Shards

A data pipeline that reproduces the data preparation strategy behind [Bagel Labs' Paris model](https://arxiv.org/abs/2510.03434) at demo scale.

Paris trains 8 independent diffusion experts in complete isolation (no gradient sync, no parameter sharing) on semantically clustered data. The paper validates at two scales: DiT-B/2 (129M params per expert, 1.03B total) and DiT-XL/2 (605M per expert, 4.84B total). This pipeline produces everything needed to train both the experts and the post-hoc router: VAE latents, CLIP text embeddings, DINOv2-clustered partitions, and router supervision data.

## Intro

I came across the Paris paper (arXiv:2510.03434) from Bagel Labs while reading about decentralized training approaches. The core idea is straightforward: take 8 independent diffusion experts (validated at DiT-B/2 and DiT-XL/2), train them on semantically clustered partitions with zero inter-node communication, then route between them at inference time with a lightweight DiT-B/2 router.

What caught my attention more than the model architecture was the data side. Paris clusters its training set using DINOv2 embeddings and conditions generation on CLIP ViT-L/14 text embeddings, with images encoded to 32x32x4 latents through Stability AI's sd-vae-ft-mse VAE. The whole thing falls apart if the clustering is bad or the preprocessing is sloppy - eight experts are only as good as the eight partitions they train on. I wanted to understand what that data pipeline actually looks like end to end, how you go from raw LAION images to clean, clustered, VAE-encoded shards ready for distributed training.

So I built one. Not at Paris scale (11M images from LAION-Aesthetic required 120 A40 GPU-days and proper orchestration), but at 10k images on a Kaggle T4 x2 instance, enough to validate every stage and hit real edge cases. The pipeline has 7 stages: ingest from `laion/laion2B-en-aesthetic` on Hugging Face (proxy for the paper's LAION-Aesthetic subset), quality filtering with NSFW detection and pHash perceptual deduplication, captioning with BLIP-2 plus CLIP ViT-L/14 text embedding extraction, DINOv2-base visual embedding followed by K-means clustering into 8 partitions, cluster validation with silhouette score and Davies-Bouldin index, VAE latent encoding to 32x32x4 float16 tensors, and finally packaging everything into per-cluster WebDataset tar shards with router training pairs.

Out of 10,000 requested images, 9,096 came back (904 dead URLs), and 8,994 survived filtering (53 flagged NSFW, 49 perceptual duplicates removed). The 8 clusters landed on recognizable semantic groupings: cars and vehicles (645 samples), dresses and fashion (544), paintings and fine art (544), illustrations and clipart (548), food and beverages (1,143), general photography as a catch-all (3,746), crafts and decorative objects (1,370), and cakes and confections (454). Silhouette score came out at 0.0425 and Davies-Bouldin at 5.0757, which is expected at this scale with the catch-all cluster absorbing 41.6% of the data. The tighter clusters showed intra-cluster cosine similarity between 0.17 and 0.30, which indicates meaningful semantic separation. The final output is 13 WebDataset shards totaling about 1.1 GB of latents, CLIP embeddings, BLIP-2 captions, and metadata, plus numpy arrays for router training. The whole run took about 38 minutes wall time.

A few places where this pipeline intentionally deviates from the paper. Paris uses DINOv2-ViT-L/14 (1024-d features) for semantic clustering (paper Section 2.6). I use DINOv2-base (768-d) as a compute tradeoff at demo scale. If you switch to ViT-L/14, you also need to update `utils/validation.py` to expect 1024-d embeddings instead of 768-d. Paris conditions on CLIP embeddings of existing LAION alt-text. I add a BLIP-2 re-captioning step (Salesforce/blip2-opt-2.7b) before CLIP encoding because LAION alt-text is noisy. Paris also uses a two-stage hierarchical clustering strategy (fine-grained k-means followed by consolidation into K=8 coarse clusters). I do single-pass K-means(K=8) here as a simplification.

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
   ├── router_training/{latents,cluster_labels,image_ids}.npy
   ├── router_training/{dinov2_embeddings,clip_text_embeddings}.npy
   └── pipeline_manifest.json
```

## Architecture Alignment

| Paris needs | This pipeline produces |
|---|---|
| 32×32×4 VAE latents per expert | Per-cluster WebDataset shards with pre-encoded latents |
| CLIP ViT-L/14 text conditioning | Pre-computed CLIP text embeddings in every shard |
| 8 semantic partitions (DINOv2) | K-means on DINOv2-base embeddings (see note on variant above), validated with silhouette score |
| Router training data | `(latent, cluster_label)` pairs exported as `router_training/latents.npy` + `router_training/cluster_labels.npy` (the Paris router adds noise/timesteps during its own training loop) |

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

> Stage 1 streams `laion/laion2B-en-aesthetic` as a proxy for the LAION-Aesthetic data used by Paris. The paper's exact 11M subset selection criteria are not publicly documented. You can adjust `laion_index_*` in `config.yaml` to change the source.

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

Silhouette of 0.0425 reflects the diversity of LAION-Aesthetic with K=8. Cluster 5 acts as a catch-all for general photography (41.6% of data), pulling the global score down. The tighter specialized clusters (0, 1, 2, 3, 7) show intra-cluster cosine similarity of 0.17–0.30, indicating meaningful semantic separation.

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

## Visualizations

### Cluster Grids

5x5 sample grids per cluster:

| | |
|---|---|
| ![Cluster 0](assets/cluster_0_grid.png) | ![Cluster 1](assets/cluster_1_grid.png) |
| ![Cluster 2](assets/cluster_2_grid.png) | ![Cluster 3](assets/cluster_3_grid.png) |
| ![Cluster 4](assets/cluster_4_grid.png) | ![Cluster 5](assets/cluster_5_grid.png) |
| ![Cluster 6](assets/cluster_6_grid.png) | ![Cluster 7](assets/cluster_7_grid.png) |

### VAE Reconstruction Verification

Side-by-side originals vs decoded latents:

![VAE reconstruction samples](assets/vae_reconstruction_samples.png)

## Shard Viewer

I built a shard viewer tool (FastAPI + React) to browse the pipeline outputs interactively. It reads the WebDataset tar files directly and provides:

- **Cluster overview dashboard** with total images, shard counts, and size distribution
- **Health checks** to spot missing, extra, or zero-size shard files
- **Filtering** by aesthetic score, cluster distance, or whether entries have a URL/caption
- **Paginated browsing** with images streamed directly from raw files (fallback to shards)
- **Entry detail modal** with larger preview, captions, scores, and source URL

This is how I originally caught leftover shards from run #1 that weren't reflected in the manifest (see "What I Learned" below). I later fixed this by making Stage 7 clean existing shard outputs by default.

![Shard viewer overview](assets/shard_viewer_overview.png)
![Shard viewer entries](assets/shard_viewer_entries.png)

## Scaling Notes

| Concern | 10k (demo) | 11M (Paris scale) |
|---------|-----------|-------------------|
| Clustering | sklearn K-means in-memory | faiss GPU K-means |
| Captioning | BLIP-2 is the slow part | If you keep re-captioning, this dominates. Paris does not require this step. |
| Storage | ~6 GB local (raw + intermediates + shards) | ~1-2 TB on object storage (rough estimate), stream via WebDataset |
| Orchestration | `run_pipeline.py` | Prefect/Airflow DAG |

**What breaks first at 11M**: sklearn K-means. 11M × 1024 float32 is about 43 GB of embeddings (assuming DINOv2-ViT-L/14 like the paper), and that is before you do anything useful with them. Use faiss.

## Interesting findings / thoughts

Building this pipeline forced me to separate two concerns: the practical mechanics of preparing data for distributed expert training, and the broader research question of whether zero-communication training is a viable paradigm for diffusion models.

### On the Paris Model's Research Claims

**The ambition is genuine, and the departure from convention is extreme.** The architecture described in the Intro (isolated experts, DINOv2 clustering, DiT backbone, post-hoc router) builds on the broader "Decentralized Diffusion Models" (DDM) paradigm, which aims to democratize diffusion training by eliminating the need for expensive, synchronized GPU clusters. However, the complete elimination of inter-expert communication represents an extreme position even within that paradigm, and raises questions about convergence guarantees and optimal data utilization.

To put the departure in perspective: Google's DiLoCo, currently among the most aggressive approaches to reducing communication overhead in distributed training, achieves roughly a 500x communication reduction and has been adopted by multiple research groups after extensive evaluation. Paris claims to eliminate communication entirely. No major AI company has adopted a similar zero-communication approach for production training. That gap could reflect conservatism and a reluctance to pursue bold departures from established methods, or it could reflect a well-founded understanding that some level of coordination is essential, a position supported by decades of optimization research and current scaling laws, which hold that performance improves predictably when compute and data scale together under properly coordinated training.

**The efficiency claims demand careful scrutiny.** The paper reports ~14.4x less training data and ~16.3x less compute compared to a prior decentralized baseline, while maintaining competitive generation quality. If verified, these would represent a paradigm-shifting advance in training efficiency. Several factors warrant skepticism:

- **Comparison baseline is narrow.** The paper references a "previous decentralized baseline" (DDM) but does not provide comprehensive benchmarks against established methods used in production by major AI labs. The efficiency gains are measured against a single, narrow reference point rather than a broad field of alternatives.
- **Evaluation scope is limited.** The assessment appears confined to FID scores without broader quality assessments or downstream task performance evaluation. FID alone is an incomplete proxy for generation quality and can mask important failure modes.
- **Independent validation is absent.** As of February 2026, this remains an arXiv preprint without peer review or independent reproduction. Meta's research teams have not reproduced or validated these results despite their potential significance. This is notable given that both Meta and Google invest heavily in communication-efficient distributed training and would have strong incentive to verify a zero-communication approach if the claims held up.
- **Tension with established scaling laws.** Scaling laws indicate that data and compute must scale together for optimal performance. A framework claiming dramatic efficiency gains while training experts on isolated data subsets needs to reconcile this tension explicitly, particularly the question of whether partitioned, uncoordinated training can match the sample efficiency of synchronized approaches.

**If communication disappears, clustering becomes the contract.** In practice, the "coordination" shows up entirely in the partitioning step: good clusters create experts with clean specialties; bad clusters strand capacity in the wrong shard, and the router cannot compensate for it after the fact. This is the single point of failure in the architecture, and the one this pipeline gave me direct experience with.

**If validated, the implications are significant.** Paris would fundamentally democratize large-scale diffusion model training by eliminating the need for centralized infrastructure. Academic institutions and smaller organizations could train competitive models using distributed commodity hardware across geographic locations. The zero-communication design could also enable multi-organization collaborations where entities contribute training compute without sharing sensitive data or requiring high-bandwidth interconnects, aligning with growing concerns around data governance and privacy.

That said, Meta's production diffusion models rely on FSDP (Fully Sharded Data Parallelism) and DDP (Distributed Data Parallelism), both of which require frequent synchronization but achieve proven scalability and quality. The broader industry's investment in *reducing* communication rather than *eliminating* it suggests that some coordination between training processes remains essential for optimal convergence. Whether Paris's zero-communication approach can match the quality of synchronized methods at production scale is an open question that rigorous, independent evaluation would need to answer. The natural next step would be an independent reproduction at small scale, benchmarked against the reported FID baselines and at least one synchronized method like DDP.

### Engineering Reality Check (What Actually Broke)

**Connection reliability.** Streaming directly from LAION kept dropping connections, so I switched to downloading 200k rows upfront and sampling from there, which made the ingest stage reliable.

**Aesthetic score metadata gap.** The aesthetic score column was missing from the initial metadata pull, which meant every shard came out with aesthetic 0.00 across the board. I fixed that by pulling the column through properly and re-running.

**Leftover shards from previous runs.** After the re-run, I noticed 4 out of 17 total shard files on disk still had zeroed aesthetic scores. These were leftover shards from the first run that still carried the old metadata. They had full data (latents, embeddings, captions), just zeroed aesthetic scores. At the time, the sharding script (`07_shard.py`) didn't clean previous `.tar` outputs before writing new ones, so the 4 leftover files coexisted with the 13 correct shards from run #2. The shard viewer's health checks are what caught this. I fixed this by making Stage 7 clean existing shard outputs by default (`clean_expert_shards: true`). There is also a possibility that some LAION records simply don't have aesthetic scores attached; to close the loop I'd either compute aesthetic scores using LAION's predictor or re-run with stricter metadata validation.

**Shard viewer as a debugging tool.** The FastAPI + React viewer (above) started as a quick debugging tool but ended up being the fastest way to audit data quality without writing throwaway scripts. It caught the leftover shard issue and made it easy to spot patterns in the cluster assignments.

Remaining work: close the aesthetic score gaps, validate data cleanliness across all shards, and publish the dataset on Hugging Face.

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

- [Paris: A Decentralized Trained Open-Weight Diffusion Model](https://arxiv.org/abs/2510.03434) - Bagel Labs
- [LAION-Aesthetic](https://laion.ai/blog/laion-aesthetics/)
- [DINOv2](https://arxiv.org/abs/2304.07193) - Meta AI
- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) - Stability AI
- [CLIP ViT-L/14](https://github.com/openai/CLIP) - OpenAI
