import os
import random
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import warnings

import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, DiffusionPipeline
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from dotenv import load_dotenv
from huggingface_hub import login, hf_hub_download
import einops

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration (can be overridden by instance config)
DEFAULT_MODEL_CONFIG = {
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

class ModelLoadingError(Exception):
    """Custom exception for model loading errors."""
    pass

class PixCellVAELoader:
    """
    A wrapper class for loading and interacting with the PixCell-1024 VAE model.
    Encapsulates model loading, device management, and core operations (sample, encode, decode).
    """
    def __init__(self, model_config=None, base_seed=42, debug=False):
        self.model_config = model_config if model_config is not None else DEFAULT_MODEL_CONFIG
        self.base_seed = base_seed
        self.debug = debug

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32 # MPS/CPU prefer fp32

        self.pipeline = None
        self.uni_model = None
        self.transform = None
        self.uncond_embedding = None
        self.caption_dims = None

        self._set_seed(self.base_seed)
        print(f"Using device: {self.device}")
        if self.debug:
            print(f"[DEBUG] Using base_seed={self.base_seed}")

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def clear_memory(self):
        """Clear GPU/CPU cache if needed."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _load_vae(self):
        print("Loading VAE...")
        vae_id = self.model_config['vae_path']
        try:
            print(f"VAE repo: {vae_id}")
            if "stable-diffusion-3.5-large" in vae_id:
                vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae", torch_dtype=self.dtype)
            else:
                vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=self.dtype)
        except Exception as e:
            print(f"VAE load failed for '{vae_id}' ({e}). Falling back to 'StonyBrook-CVLab/sd-vae-ft-ema-path' (4-ch).")
            vae = AutoencoderKL.from_pretrained("StonyBrook-CVLab/sd-vae-ft-ema-path", torch_dtype=self.dtype)

        try:
            lc = getattr(vae.config, 'latent_channels', None)
            print(f"VAE latent_channels: {lc}")
        except Exception:
            pass
        return vae

    def _load_pipeline(self, vae):
        print("Loading PixCell pipeline...")
        pipeline = DiffusionPipeline.from_pretrained(
            self.model_config['pipeline_path'],
            vae=vae,
            custom_pipeline=self.model_config['pipeline_name'],
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        # Memory saving knobs
        try:
            pipeline.enable_attention_slicing()
            pipeline.enable_vae_slicing()
            if self.device.type == 'cuda':
                pipeline.enable_sequential_cpu_offload()
            else:
                pipeline.to(self.device, dtype=self.dtype)
        except Exception:
            pipeline.to(self.device, dtype=self.dtype)
        # Patch: some VAEs have shift_factor=None
        if getattr(pipeline.vae.config, 'shift_factor', None) is None:
            pipeline.vae.config.shift_factor = 0.0
        pipeline.vae.to(self.device, dtype=self.dtype)
        # Sanity: warn if VAE latent_channels != 16 (PixCell uses SD3-family 16-ch latents)
        try:
            lc = getattr(pipeline.vae.config, 'latent_channels', None)
            if lc is not None and lc != 16:
                print(f"[WARN] VAE latent_channels={lc}. PixCell expects 16 (SD3 VAE). You may hit decode errors.")
        except Exception:
            pass
        # Ensure components are on the intended device/dtype for MPS/CPU
        if self.device.type != 'cuda':
            pipeline.to(self.device, dtype=self.dtype)
        print("PixCell pipeline loaded successfully")
        return pipeline

    def _load_uni_model(self):
        """Load and return the UNI model and its transform."""
        print("Loading UNI model...")
        uni_model = timm.create_model(
            self.model_config['uni_model_name'],
            pretrained=True,
            **self.model_config['uni_model_config']
        )
        uni_model.eval()
        uni_model.to(self.device)
        transform = create_transform(**resolve_data_config(uni_model.pretrained_cfg, model=uni_model))
        return uni_model, transform

    def load_models(self):
        """Load all required models and return them in a dictionary."""
        # Prioritize HF_TOKEN as it seems to be the one causing the conflict
        token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
        if not token:
            raise ModelLoadingError(
                "Hugging Face token not found. Set HF_TOKEN (preferred) or HUGGING_FACE_HUB_TOKEN in your .env."
            )
        
        try:
            login(token=token)
            
            vae = self._load_vae()
            self.pipeline = self._load_pipeline(vae)
            self.uni_model, self.transform = self._load_uni_model()
            
            self.caption_dims = (
                self.pipeline.transformer.config.caption_num_tokens,
                self.pipeline.transformer.config.caption_channels
            )
            if self.debug:
                print(f"[DEBUG] caption_num_tokens={self.caption_dims[0]}, caption_channels={self.caption_dims[1]}")
            
            print(f"Model expects UNI embeddings with shape: (batch_size, {self.caption_dims[0]}, {self.caption_dims[1]})")
            
            self.uncond_embedding = self.pipeline.get_unconditional_embedding(1).to(self.device)
            
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            raise ModelLoadingError(f"Failed to load models: {e}. Check token, internet, and model paths.") from e

    def _fix_token_count(self, emb: torch.Tensor, target_tokens: int) -> torch.Tensor:
        """Ensure the UNI embedding has exactly (B, target_tokens, D). Trim or tile as needed."""
        B, T, D = emb.shape
        if T == target_tokens:
            return emb
        if T > target_tokens:
            return emb[:, :target_tokens, :]
        reps = (target_tokens + T - 1) // T
        emb_tiled = emb.repeat(1, reps, 1)[:, :target_tokens, :]
        return emb_tiled

    def _load_uni_from_dir(self, tiles_dir: str, max_tiles: int = 16) -> torch.Tensor:
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
                if min(im.size) < 64:
                    continue
                batch.append(self.transform(im))
                kept += 1
            except Exception:
                continue
        if kept == 0:
            raise FileNotFoundError(f"No readable RGB images found in {tiles_dir} after filtering.")
        return torch.stack(batch, dim=0)

    def _download_and_process_image(self):
        """Fallback: download and process the example image for conditioning (used if CONDITION_DIR not set)."""
        path = hf_hub_download(
            repo_id=self.model_config['pipeline_path'],
            filename="test_image.png"
        )
        image = Image.open(path).convert("RGB")
        
        uni_patches = np.array(image)
        uni_patches = einops.rearrange(
            uni_patches, 
            '(d1 h) (d2 w) c -> (d1 d2) h w c', 
            d1=4, d2=4
        )
        return torch.stack([self.transform(Image.fromarray(item)) for item in uni_patches])

    def _get_conditioning_batch(self, cond_dir: str = "") -> torch.Tensor:
        """Return a batch for UNI: CLI --cond_dir > ENV CONDITION_DIR > HF fallback."""
        if cond_dir:
            print(f"Using local conditioning tiles from: {cond_dir}")
            batch = self._load_uni_from_dir(cond_dir, max_tiles=16)
            print(f"Loaded {batch.shape[0]} tiles for UNI conditioning.")
            return batch
        else:
            print("No cond_dir provided; using bundled HF example image for conditioning")
            return self._download_and_process_image()

    def _extract_uni_embeddings(self, image_batch: torch.Tensor):
        """Extract UNI embeddings from a batch of transformed images.
        Args:
            image_batch: Tensor of shape (N, C, H, W) already transformed
        Returns:
            Tensor of shape (1, N, D) on the active device
        """
        with torch.no_grad():
            feats = self.uni_model(image_batch.to(self.device))  # (N, D)
            if feats.dim() == 2:
                feats = feats.unsqueeze(0)             # (1, N, D)
        return feats.to(self.device)

    def sample(self, num_samples: int = None, cond_dir: str = "") -> list[Image.Image]:
        """
        Generate synthetic images using the PixCell pipeline.
        Args:
            num_samples: Number of samples to generate (uses config if None)
            cond_dir: Optional directory for conditioning images.
        Returns:
            A list of PIL Image objects.
        """
        if self.pipeline is None or self.uni_model is None or self.transform is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        print("\nStarting generation...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        num_samples = num_samples if num_samples is not None else self.model_config['generation']['num_samples']
        
        try:
            image_input = self._get_conditioning_batch(cond_dir)
            print(f"Conditioning batch: {tuple(image_input.shape)}")

            uni_emb = self._extract_uni_embeddings(image_input)
            if self.debug:
                print(f"[DEBUG] uni_emb raw shape: {tuple(uni_emb.shape)} | mean={uni_emb.mean().item():.4f} std={uni_emb.std().item():.4f}")

            target_tokens = self.caption_dims[0]
            if uni_emb.shape[1] != target_tokens:
                if target_tokens == 1:
                    uni_emb = uni_emb.mean(dim=1, keepdim=True)
                else:
                    uni_emb = self._fix_token_count(uni_emb, target_tokens)
            print(f"UNI embeddings shaped for PixCell: {tuple(uni_emb.shape)} (target tokens={target_tokens})")

            uncond = self.pipeline.get_unconditional_embedding(uni_emb.shape[0]).to(self.device)
            
            samples = []
            gen_config = self.model_config['generation']
            use_autocast = 'cuda' in str(self.device)
            
            for i in range(num_samples):
                print(f"\nGenerating sample {i+1}/{num_samples}")
                try:
                    generator = torch.Generator("cpu").manual_seed(self.base_seed + i)
                    
                    if self.debug:
                        print(f"[DEBUG] call: steps={gen_config['num_inference_steps']} guidance={gen_config['guidance_scale']} uni.shape={tuple(uni_emb.shape)} uncond.shape={tuple(uncond.shape)}")
                    
                    if use_autocast:
                        with torch.amp.autocast(device_type='cuda'):
                            sample = self.pipeline(
                                uni_embeds=uni_emb,
                                negative_uni_embeds=uncond,
                                guidance_scale=gen_config['guidance_scale'],
                                generator=generator,
                                num_inference_steps=gen_config['num_inference_steps']
                            )
                    else:
                        sample = self.pipeline(
                            uni_embeds=uni_emb,
                            negative_uni_embeds=uncond,
                            guidance_scale=gen_config['guidance_scale'],
                            generator=generator,
                            num_inference_steps=gen_config['num_inference_steps']
                        )
                    samples.append(sample.images[0])
                    
                except Exception as e:
                    print(f"Error generating sample {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
            
            return samples
                
        except Exception as e:
            print(f"Error during sample generation: {e}")
            import traceback
            traceback.print_exc()
            return []

    def encode(self, image: Image.Image) -> torch.Tensor:
        """Encode image to latent representation."""
        if self.uni_model is None or self.transform is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        img_array = np.array(image)
        
        patches = einops.rearrange(
            img_array,
            '(d1 h) (d2 w) c -> (d1 d2) h w c',
            d1=4, d2=4
        )
        
        patch_tensors = torch.stack([
            self.transform(Image.fromarray(patch)) 
            for patch in patches
        ])
        
        with torch.inference_mode():
            # Extract UNI embeddings, which returns (1, num_patches, D)
            uni_emb = self._extract_uni_embeddings(patch_tensors)
            
            target_tokens = self.caption_dims[0]
            if uni_emb.shape[1] != target_tokens:
                if target_tokens == 1:
                    uni_emb = uni_emb.mean(dim=1, keepdim=True)
                else:
                    uni_emb = self._fix_token_count(uni_emb, target_tokens)
            
        return uni_emb
    
    def decode(self, latent: torch.Tensor) -> Image.Image:
        """Decode latent representation back to image."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_models() first.")
            
        uncond = self.pipeline.get_unconditional_embedding(latent.shape[0])
        
        with torch.amp.autocast(device_type=self.device.type if self.device.type == 'cuda' else 'cpu'): # MPS doesn't use autocast
            result = self.pipeline(
                uni_embeds=latent,
                negative_uni_embeds=uncond,
                guidance_scale=self.model_config['generation']['guidance_scale'],
                num_inference_steps=self.model_config['generation']['num_inference_steps']
            )
            
        return result.images[0]