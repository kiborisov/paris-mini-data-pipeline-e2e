"""Stage 3: Synthetic captioning (BLIP-2) and CLIP text embedding extraction.

Produces two outputs:
- Synthetic captions for each image (human-readable, stored in parquet)
- CLIP ViT-L/14 text embeddings (the actual conditioning vectors Paris uses)

GPU required.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.helpers import (
    get_device,
    load_checkpoint,
    load_image,
    load_metadata,
    require_gpu,
    save_checkpoint,
    save_metadata,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Captioning
# ---------------------------------------------------------------------------

def _load_caption_model(config: dict):
    """Load BLIP-2 captioning model. Falls back to BLIP-large if OOM."""
    device = get_device()

    try:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        model_name = config["caption_model"]
        logger.info(f"Loading {model_name}...")

        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        model.eval()

        return processor, model, model_name

    except (RuntimeError, torch.cuda.OutOfMemoryError):
        fallback = config.get("caption_model_fallback", "Salesforce/blip-image-captioning-large")
        logger.warning(f"BLIP-2 OOM — falling back to {fallback}")

        from transformers import BlipForConditionalGeneration, BlipProcessor

        processor = BlipProcessor.from_pretrained(fallback)
        model = BlipForConditionalGeneration.from_pretrained(
            fallback, torch_dtype=torch.float16
        ).to(device)
        model.eval()

        return processor, model, fallback


def _caption_images(
    image_paths: list[str],
    config: dict,
) -> list[str]:
    """Generate synthetic captions for all images."""
    device = get_device()
    batch_size = config["caption_batch_size"]
    max_tokens = config["caption_max_tokens"]
    checkpoint_interval = config["caption_checkpoint_interval"]

    # Check for existing checkpoint
    checkpoint = load_checkpoint("caption", config["checkpoint_dir"])
    start_idx = 0
    captions = []
    if checkpoint:
        captions = checkpoint["captions"]
        start_idx = checkpoint["next_idx"]
        logger.info(f"Resuming captioning from index {start_idx} ({len(captions)} done)")

    processor, model, model_name = _load_caption_model(config)
    logger.info(f"Using captioning model: {model_name}")

    for i in tqdm(range(start_idx, len(image_paths), batch_size), desc="Captioning"):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            try:
                images.append(load_image(p))
            except Exception:
                images.append(load_image(batch_paths[0]))  # placeholder

        inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

        batch_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        captions.extend([c.strip() for c in batch_captions])

        # Checkpoint
        if len(captions) % checkpoint_interval < batch_size:
            save_checkpoint(
                {"captions": captions, "next_idx": i + batch_size},
                "caption",
                config["checkpoint_dir"],
            )

    # Clean up GPU memory
    del model, processor
    torch.cuda.empty_cache()

    return captions


# ---------------------------------------------------------------------------
# CLIP text embeddings
# ---------------------------------------------------------------------------

def _encode_clip_text(captions: list[str], config: dict) -> np.ndarray:
    """Encode captions to CLIP ViT-L/14 text embeddings.

    Returns: (N, 77, 768) float16 array.
    """
    from transformers import CLIPTextModel, CLIPTokenizer

    device = get_device()
    model_name = config["clip_model"]
    max_length = config["clip_max_length"]

    logger.info(f"Loading CLIP text model: {model_name}...")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_model = CLIPTextModel.from_pretrained(model_name).to(device)
    text_model.eval()

    all_embeddings = []
    batch_size = 32  # CLIP text encoding is lightweight

    for i in tqdm(range(0, len(captions), batch_size), desc="CLIP encoding"):
        batch = captions[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = text_model(**inputs)
            # Full sequence embeddings for conditioning (not just pooled)
            embeddings = outputs.last_hidden_state.cpu().numpy().astype(np.float16)

        all_embeddings.append(embeddings)

    del text_model, tokenizer
    torch.cuda.empty_cache()

    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config: dict) -> dict:
    """Run captioning and CLIP text embedding extraction.

    Returns:
        Dict with stage metrics.
    """
    require_gpu()

    df = load_metadata(str(Path(config["filtered_data_dir"]) / "metadata.parquet"))
    image_paths = df["image_path"].tolist()

    # Step 1: Generate captions
    logger.info(f"Captioning {len(image_paths)} images...")
    captions = _caption_images(image_paths, config)
    df["synthetic_caption"] = captions

    # Spot-check: log 5 random caption-image pairs
    import random
    sample_indices = random.sample(range(len(df)), min(5, len(df)))
    for idx in sample_indices:
        logger.info(
            f"  Sample caption: '{captions[idx]}' "
            f"(image: {Path(image_paths[idx]).name})"
        )

    # Step 2: CLIP text embeddings
    logger.info("Extracting CLIP text embeddings...")
    clip_embeddings = _encode_clip_text(captions, config)

    # Save outputs
    output_dir = Path(config["captioned_data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    save_metadata(df, str(output_dir / "metadata.parquet"))

    emb_path = str(output_dir / "clip_text_embeddings.npy")
    np.save(emb_path, clip_embeddings)
    logger.info(f"CLIP embeddings saved: {clip_embeddings.shape} to {emb_path}")

    # Clear checkpoint on success
    from utils.helpers import clear_checkpoint
    clear_checkpoint("caption", config["checkpoint_dir"])

    return {
        "images_captioned": len(captions),
        "clip_embedding_shape": list(clip_embeddings.shape),
        "caption_model": config["caption_model"],
        "clip_model": config["clip_model"],
    }
