import os

"""
Optimized PixCell Inference Script

This script provides memory-efficient inference for the PixCell model by:
1. Processing one sample at a time
2. Using gradient checkpointing
3. Managing GPU memory with cache clearing
4. Using torch.no_grad() for inference
"""

# Optional: lower MPS high watermark to release memory more aggressively (harmless on non-MPS)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# Standard library imports
import random
from pathlib import Path
from datetime import datetime
import warnings


import argparse

# ---- Seeding helper ---------------------------------------------------------
from datetime import datetime  # already imported above; safe to re-import

def set_seed(seed: int):
    import random as _random
    import numpy as _np
    import torch as _torch
    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
# ----------------------------------------------------------------------------

DEBUG = os.getenv("PIX_DEBUG", "0").strip() == "1"

# Third-party imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import AutoencoderKL, DiffusionPipeline
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from dotenv import load_dotenv
from huggingface_hub import login, hf_hub_download
import einops

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
SEED = 34
MODEL_CONFIG = {
    'vae_path': "stabilityai/stable-diffusion-3.5-large",
    'pipeline_path': "StonyBrook-CVLab/PixCell-1024",
    'pipeline_name': "StonyBrook-CVLab/PixCell-pipeline",
    'uni_model_name': "hf-hub:MahmoodLab/UNI2-h",
    'uni_model_config': {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    },
    'generation': {
        'num_inference_steps': 22,
        'guidance_scale': 1.5,
        'num_samples': 2
    }
}

# Set random seeds for reproducibility
BASE_SEED = int(os.getenv("PIX_SEED", SEED))
set_seed(BASE_SEED)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
DTYPE = torch.float16 if device.type == 'cuda' else torch.float32  # MPS/CPU prefer fp32

# Prefer the 256px model on MPS/CPU for speed; keep 1024 only on CUDA
if device.type != 'cuda' and MODEL_CONFIG.get('pipeline_path', '').endswith('PixCell-1024'):
    print("MPS/CPU detected: switching to PixCell-256 for faster inference.")
    MODEL_CONFIG['pipeline_path'] = "StonyBrook-CVLab/PixCell-256"
    # lighter defaults for non-CUDA
    MODEL_CONFIG['generation']['num_inference_steps'] = min(MODEL_CONFIG['generation']['num_inference_steps'], 18)
    MODEL_CONFIG['generation']['guidance_scale'] = 1.0

# Output directory
OUT_DIR = Path("generated_samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class ModelLoadingError(Exception):
    """Custom exception for model loading errors."""
    pass

def clear_memory():
    """Clear GPU/CPU cache if needed."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _load_vae():
    print("Loading VAE...")
    vae_id = MODEL_CONFIG['vae_path']
    try:
        print(f"VAE repo: {vae_id}")
        if "stable-diffusion-3.5-large" in vae_id:
            vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae", torch_dtype=DTYPE)
        else:
            vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=DTYPE)
    except Exception as e:
        print(f"VAE load failed for '{vae_id}' ({e}). Falling back to 'StonyBrook-CVLab/sd-vae-ft-ema-path' (4-ch).")
        vae = AutoencoderKL.from_pretrained("StonyBrook-CVLab/sd-vae-ft-ema-path", torch_dtype=DTYPE)

    # Report latent channels so we can quickly spot 4-ch vs 16-ch
    try:
        lc = getattr(vae.config, 'latent_channels', None)
        print(f"VAE latent_channels: {lc}")
    except Exception:
        pass
    return vae


def _load_pipeline(vae):
    print("Loading PixCell pipeline...")
    pipeline = DiffusionPipeline.from_pretrained(
        MODEL_CONFIG['pipeline_path'],
        vae=vae,
        custom_pipeline=MODEL_CONFIG['pipeline_name'],
        trust_remote_code=True,
        torch_dtype=DTYPE,
    )
    # Memory saving knobs
    try:
        pipeline.enable_attention_slicing()
        pipeline.enable_vae_slicing()
        if device.type == 'cuda':
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline.to(device, dtype=DTYPE)
    except Exception:
        pipeline.to(device, dtype=DTYPE)
    # Patch: some VAEs have shift_factor=None
    if getattr(pipeline.vae.config, 'shift_factor', None) is None:
        pipeline.vae.config.shift_factor = 0.0
    pipeline.vae.to(device, dtype=DTYPE)
    # Sanity: warn if VAE latent_channels != 16 (PixCell uses SD3-family 16-ch latents)
    try:
        lc = getattr(pipeline.vae.config, 'latent_channels', None)
        if lc is not None and lc != 16:
            print(f"[WARN] VAE latent_channels={lc}. PixCell expects 16 (SD3 VAE). You may hit decode errors.")
    except Exception:
        pass
    # Ensure components are on the intended device/dtype for MPS/CPU
    if device.type != 'cuda':
        pipeline.to(device, dtype=DTYPE)
    print("PixCell pipeline loaded successfully")
    return pipeline


def _load_uni_model():
    """Load and return the UNI model and its transform."""
    print("Loading UNI model...")
    uni_model = timm.create_model(
        MODEL_CONFIG['uni_model_name'],
        pretrained=True,
        **MODEL_CONFIG['uni_model_config']
    )
    uni_model.eval()
    uni_model.to(device)
    transform = create_transform(**resolve_data_config(uni_model.pretrained_cfg, model=uni_model))
    return uni_model, transform


def load_models():
    """Load all required models and return them in a dictionary."""
    # Accept either HUGGING_FACE_HUB_TOKEN or HF_TOKEN from .env / environment
    token = os.getenv('HUGGING_FACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
    if not token:
        raise ModelLoadingError(
            "Hugging Face token not found. Set HUGGING_FACE_HUB_TOKEN (preferred) or HF_TOKEN in your .env."
        )
    
    try:
        # Login to Hugging Face
        login(token=token)
        
        # Load models
        vae = _load_vae()
        pipeline = _load_pipeline(vae)
        uni_model, transform = _load_uni_model()
        
        # Get model dimensions
        caption_num_tokens = pipeline.transformer.config.caption_num_tokens
        caption_channels = pipeline.transformer.config.caption_channels
        if DEBUG:
            print(f"[DEBUG] caption_num_tokens={caption_num_tokens}, caption_channels={caption_channels}")
        # Optional override for experiments
        override_tokens = None
        try:
            import sys
            for i, a in enumerate(sys.argv):
                if a == "--tokens" and i+1 < len(sys.argv):
                    override_tokens = int(sys.argv[i+1])
                    break
        except Exception:
            pass
        if override_tokens is not None:
            print(f"[DEBUG] Overriding caption_num_tokens -> {override_tokens}")
            pipeline.transformer.config.caption_num_tokens = override_tokens
            caption_num_tokens = override_tokens
        print(f"Model expects UNI embeddings with shape: (batch_size, {caption_num_tokens}, {caption_channels})")
        
        # Generate unconditional embedding
        print("Generating unconditional embedding...")
        uncond = pipeline.get_unconditional_embedding(1).to(device)
        
        return {
            'pipeline': pipeline,
            'uni_model': uni_model,
            'transform': transform,
            'uncond': uncond,
            'dims': (caption_num_tokens, caption_channels)
        }
        
    except Exception as e:
        raise ModelLoadingError(f"Failed to load models: {e}") from e

# ---------------- UNI conditioning helpers ----------------
from glob import glob

def _fix_token_count(emb: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Ensure the UNI embedding has exactly (B, target_tokens, D). Trim or tile as needed."""
    B, T, D = emb.shape
    if T == target_tokens:
        return emb
    if T > target_tokens:
        return emb[:, :target_tokens, :]
    # T < target_tokens: tile to reach target_tokens
    reps = (target_tokens + T - 1) // T
    emb_tiled = emb.repeat(1, reps, 1)[:, :target_tokens, :]
    return emb_tiled

def _load_uni_from_dir(transform, tiles_dir: str, max_tiles: int = 16) -> torch.Tensor:
    """Recursively load up to `max_tiles` images from `tiles_dir`.
    - Supports extensions: png, jpg, jpeg, tif, tiff (case-insensitive)
    - Searches subfolders
    - Skips unreadable / tiny files
    Returns a tensor of shape (N, C, H, W).
    """
    allowed = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    paths = []
    for root, _, files in os.walk(tiles_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in allowed:
                paths.append(os.path.join(root, fn))
    if not paths:
        raise FileNotFoundError(f"No images found in {tiles_dir} (looked for {sorted(allowed)})")

    # Sort for determinism and cap to max_tiles
    paths = sorted(paths)[:max_tiles]
    print(f"Found {len(paths)} conditioning tiles (showing up to {max_tiles}).")
    for p in paths[:5]:
        print(f"  - {p}")

    batch = []
    kept = 0
    for p in paths:
        try:
            im = Image.open(p)
            im = im.convert("RGB")
            # quick size sanity (skip tiny icons by accident)
            if min(im.size) < 64:
                continue
            batch.append(transform(im))
            kept += 1
        except Exception:
            continue
    if kept == 0:
        raise FileNotFoundError(f"No readable RGB images found in {tiles_dir} after filtering.")
    return torch.stack(batch, dim=0)

def _get_conditioning_batch(transform) -> torch.Tensor:
    """Return a batch for UNI: CLI --cond_dir > ENV CONDITION_DIR > HF fallback."""
    # CLI override
    import sys
    try:
        if "--cond_dir" in sys.argv:
            idx = sys.argv.index("--cond_dir")
            if idx+1 < len(sys.argv):
                cond_dir = sys.argv[idx+1]
            else:
                cond_dir = ""
        else:
            cond_dir = os.getenv("CONDITION_DIR", "").strip()
    except Exception:
        cond_dir = os.getenv("CONDITION_DIR", "").strip()
    if cond_dir:
        print(f"Using local conditioning tiles from: {cond_dir}")
        batch = _load_uni_from_dir(transform, cond_dir, max_tiles=16)
        print(f"Loaded {batch.shape[0]} tiles for UNI conditioning.")
        return batch
    else:
        print("No cond_dir provided; using bundled HF example image for conditioning")
        return _download_and_process_image(transform)
# ---------------------------------------------------------

def _download_and_process_image(transform):
    """Fallback: download and process the example image for conditioning (used if CONDITION_DIR not set)."""
    path = hf_hub_download(
        repo_id=MODEL_CONFIG['pipeline_path'],
        filename="test_image.png"
    )
    image = Image.open(path).convert("RGB")
    
    # Prepare image patches for UNI model
    uni_patches = np.array(image)
    uni_patches = einops.rearrange(
        uni_patches, 
        '(d1 h) (d2 w) c -> (d1 d2) h w c', 
        d1=4, d2=4  # Split 1024x1024 into 16x 256x256 patches
    )
    return torch.stack([transform(Image.fromarray(item)) for item in uni_patches])


def _extract_uni_embeddings(uni_model, transform, image_batch: torch.Tensor):
    """Extract UNI embeddings from a batch of transformed images.
    Args:
        uni_model: UNI-2h encoder (in eval mode, on device)
        transform: (unused here; kept for API compatibility)
        image_batch: Tensor of shape (N, C, H, W) already transformed
    Returns:
        Tensor of shape (1, N, D) on the active device
    """
    with torch.no_grad():
        feats = uni_model(image_batch.to(device))  # (N, D)
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)             # (1, N, D)
    return feats.to(device)


def _generate_samples_impl(pipeline, uni_emb, uncond, num_samples, timestamp, base_seed: int):
    """Core implementation of sample generation."""
    samples = []
    gen_config = MODEL_CONFIG['generation']
    
    # Determine device type for autocast
    use_autocast = 'cuda' in str(device)
    
    for i in range(num_samples):
        print(f"\nGenerating sample {i+1}/{num_samples}")
        try:
            generator = torch.Generator("cpu").manual_seed(base_seed + i)
            
            # NOTE: PixCell will raise: "Number of UNI embeddings must match the ones used in training (1)"
            # if we pass more than caption_num_tokens tokens. The pooling/trim above guarantees compliance.
            # Only use autocast for CUDA devices
            if DEBUG:
                print(f"[DEBUG] call: steps={gen_config['num_inference_steps']} guidance={gen_config['guidance_scale']} uni.shape={tuple(uni_emb.shape)} uncond.shape={tuple(uncond.shape)}")
            if use_autocast:
                with torch.amp.autocast(device_type='cuda'):
                    sample = pipeline(
                        uni_embeds=uni_emb,
                        negative_uni_embeds=uncond,
                        guidance_scale=gen_config['guidance_scale'],
                        generator=generator,
                        num_inference_steps=gen_config['num_inference_steps']
                    )
            else:
                # For MPS/CPU, run without autocast
                sample = pipeline(
                    uni_embeds=uni_emb,
                    negative_uni_embeds=uncond,
                    guidance_scale=gen_config['guidance_scale'],
                    generator=generator,
                    num_inference_steps=gen_config['num_inference_steps']
                )
            
            # Save the generated image
            img_path = OUT_DIR / f'sample_{timestamp}_{i:02d}.png'
            sample.images[0].save(img_path)
            print(f"Saved to {img_path}")
            samples.append(sample.images[0])
            
        except Exception as e:
            print(f"Error generating sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    return samples


def generate_samples(pipeline, num_samples=None, uni_model=None, transform=None, uncond=None):
    """
    Generate samples using the PixCell pipeline with UNI model.
    
    Args:
        pipeline: Loaded PixCell pipeline
        num_samples: Number of samples to generate (uses config if None)
        uni_model: Loaded UNI model
        transform: Image transform for UNI model
        uncond: Unconditional embedding for guidance
    """
    if None in [pipeline, uni_model, transform, uncond]:
        raise ValueError("All model components must be provided")
    
    print("\nStarting generation...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    num_samples = num_samples or MODEL_CONFIG['generation']['num_samples']
    
    try:
        # Build conditioning batch (local tiles preferred via CONDITION_DIR)
        image_input = _get_conditioning_batch(transform)
        print(f"Conditioning batch: {tuple(image_input.shape)}")

        # Extract UNI embeddings
        uni_emb = _extract_uni_embeddings(uni_model, transform, image_input)  # (1, T, D)

        if DEBUG:
            print(f"[DEBUG] uni_emb raw shape: {tuple(uni_emb.shape)} | mean={uni_emb.mean().item():.4f} std={uni_emb.std().item():.4f}")

        # Match PixCell's expected caption token count
        target_tokens = getattr(pipeline.transformer.config, 'caption_num_tokens', 1)
        if uni_emb.shape[1] != target_tokens:
            if target_tokens == 1:
                # Average-pool tokens to a single conditioning token
                uni_emb = uni_emb.mean(dim=1, keepdim=True)  # (1, 1, D)
            else:
                # Trim or tile to the required number of tokens
                uni_emb = _fix_token_count(uni_emb, target_tokens)
        print(f"UNI embeddings shaped for PixCell: {tuple(uni_emb.shape)} (target tokens={target_tokens})")

        # Get matching-size unconditional embedding for classifier-free guidance
        uncond = pipeline.get_unconditional_embedding(uni_emb.shape[0]).to(device)

        return _generate_samples_impl(
            pipeline, uni_emb, uncond, num_samples, timestamp,
            base_seed=int(os.getenv("PIX_SEED", BASE_SEED))
        )
            
    except Exception as e:
        print(f"Error during sample generation: {e}")
        import traceback
        traceback.print_exc()
        return []


# ----------- Helper for unconditional control image -----------
def run_uncond_control(pipeline, uncond, timestamp):
    gen_config = MODEL_CONFIG['generation']
    generator = torch.Generator("cpu").manual_seed(SEED + 999)
    sample = pipeline(
        uni_embeds=uncond,               # <- use uncond as the conditioner
        negative_uni_embeds=uncond,
        guidance_scale=gen_config['guidance_scale'],
        generator=generator,
        num_inference_steps=gen_config['num_inference_steps']
    )
    img_path = OUT_DIR / f'control_uncond_{timestamp}.png'
    sample.images[0].save(img_path)
    print(f"Saved uncond-control to {img_path}")

def display_samples(samples):
    """Display generated samples in a grid."""
    if not samples:
        print("No samples to display")
        return
        
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
    
    if n == 1:
        axes = [axes]
        
    for i, img in enumerate(samples):
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')
        
    plt.tight_layout()
    plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cond_dir", type=str, default=os.getenv("CONDITION_DIR", ""))
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--tokens", type=int, default=None, help="Override caption_num_tokens (debug)")
    p.add_argument("--uncond_test", action="store_true", help="Run a control sample using unconditional embedding as the conditioner")
    p.add_argument("--fast", action="store_true", help="Use very light settings (steps=12, guidance=1.0)")
    p.add_argument("--num", type=int, default=None, help="Number of images to generate (overrides config)")
    p.add_argument("--seed", type=int, default=None, help="Base seed for this run (overrides PIX_SEED & default)")
    p.add_argument("--randomize", action="store_true", help="Use a time-based random seed for this run")
    return p.parse_args()


def main():
    """Main function to run the PixCell inference pipeline."""
    print(f"Using device: {device}")

    args = parse_args()
    if args.fast:
        MODEL_CONFIG['generation']['num_inference_steps'] = 12
        MODEL_CONFIG['generation']['guidance_scale'] = 1.0
    if args.steps is not None:
        MODEL_CONFIG['generation']['num_inference_steps'] = int(args.steps)
    if args.guidance is not None:
        MODEL_CONFIG['generation']['guidance_scale'] = float(args.guidance)

    # Override number of images if provided
    if args.num is not None:
        MODEL_CONFIG['generation']['num_samples'] = int(args.num)

    # Decide run seed: --seed > --randomize > PIX_SEED env > default
    run_seed = BASE_SEED
    if args.seed is not None:
        run_seed = int(args.seed)
    elif args.randomize:
        run_seed = int(datetime.now().timestamp())
    set_seed(run_seed)
    os.environ["PIX_SEED"] = str(run_seed)
    if DEBUG:
        print(f"[DEBUG] Using run_seed={run_seed}")

    # Echo which token env is set (helps debug auth issues)
    if os.getenv('HUGGING_FACE_HUB_TOKEN'):
        print("Auth: using HUGGING_FACE_HUB_TOKEN from environment/.env")
    elif os.getenv('HF_TOKEN'):
        print("Auth: using HF_TOKEN from environment/.env")

    try:
        # Load models
        print("Initializing models...")
        models = load_models()

        # Set conditioning dir from CLI arg if provided
        cond_dir = args.cond_dir.strip() if args.cond_dir else os.getenv("CONDITION_DIR", "").strip()
        if cond_dir:
            print(f"Conditioning directory set to: {cond_dir}")
            os.environ["CONDITION_DIR"] = cond_dir

        # Generate samples
        print("\nGenerating samples...")
        samples = generate_samples(
            pipeline=models['pipeline'],
            num_samples=MODEL_CONFIG['generation']['num_samples'],
            uni_model=models['uni_model'],
            transform=models['transform'],
            uncond=models['uncond']
        )

        # Optionally run unconditional control branch
        if args.uncond_test:
            run_uncond_control(models['pipeline'], models['uncond'], datetime.now().strftime('%Y%m%d_%H%M%S'))

        # Display results
        if samples:
            print(f"\nSuccessfully generated {len(samples)} samples")
            display_samples(samples)
        else:
            print("\nNo samples were generated.")

    except ModelLoadingError as e:
        print(f"\nError: {e}")
        print("Please check your Hugging Face token and internet connection.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if 'models' in locals() and 'pipeline' in models:
            del models['pipeline']
        clear_memory()
        print("\nDone!")


if __name__ == "__main__":
    main()