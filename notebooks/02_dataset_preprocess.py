#!/usr/bin/env python3
"""
02 — Dataset Preparation and Reconstruction
-----------------------------------------
Enhanced version with command-line controls for processing histopathology images.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
import requests
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download
from diffusers import AutoencoderKL, DiffusionPipeline
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import einops
import shutil
import random
from datetime import datetime

try:
    import kagglehub
except ImportError:
    kagglehub = None

# Setup logging
def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

# Get the project root directory (one level up from notebooks/)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

class Config:
    """Configuration class with default values that can be overridden by command line args."""
    def __init__(self, args):
        # Input/Output
        self.input_dir = Path(args.input_dir) if args.input_dir else PROJECT_ROOT / 'data' / 'inputs'
        self.output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / 'data' / 'outputs' / 'reconstructions'
        
        # Model parameters
        self.image_size = args.image_size
        self.device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Dataset settings
        self.kaggle_dataset = args.kaggle_dataset
        self.max_samples = args.max_samples
        self.skip_download = args.skip_download
        
        # Model parameters
        self.model_name = "StonyBrook-CVLab/PixCell-1024"
        self.vae_name = "stabilityai/stable-diffusion-3.5-large"
        self.uni_model_name = "hf-hub:MahmoodLab/UNI2-h"
        
        # Generation parameters
        self.guidance_scale = args.guidance_scale
        self.num_inference_steps = args.steps
        
        # Random seed
        self.seed = args.seed if args.seed is not None else 42
        
        # Ensure directories exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process histopathology images with PixCell model.')
    
    # I/O parameters
    parser.add_argument('--input-dir', type=str, help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, help='Directory to save processed images')
    
    # Dataset parameters
    parser.add_argument('--kaggle-dataset', type=str, 
                       default='sani84/glasmiccai2015-gland-segmentation',
                       help='Kaggle dataset identifier')
    parser.add_argument('--max-samples', type=int, default=12,
                       help='Maximum number of samples to process')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip downloading from Kaggle')
    
    # Processing parameters
    parser.add_argument('--image-size', type=int, default=1024,
                       help='Size to resize images to (square)')
    parser.add_argument('--steps', type=int, default=15,
                       help='Number of inference steps')
    parser.add_argument('--guidance-scale', type=float, default=1.5,
                       help='Guidance scale for generation')
    
    # Control parameters
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_sample_images(config, logger):
    """Download sample histopathology images."""
    if config.skip_download:
        logger.info("Skipping download as requested")
        return
        
    logger.info(f"Preparing sample histology images (max {config.max_samples})...")
    os.makedirs(config.input_dir, exist_ok=True)

    def _copy_from_path(root):
        count = 0
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
        for pattern in exts:
            for src in Path(root).rglob(pattern):
                if count >= config.max_samples:
                    return count
                try:
                    with Image.open(src) as im:
                        im = im.convert("RGB")
                        dest = config.input_dir / f"sample_{count+1}.png"
                        im.save(dest)
                        count += 1
                        logger.debug(f"Copied: {src} -> {dest}")
                except Exception as e:
                    logger.debug(f"Skip file {src}: {e}")
                    continue
        return count

    # Try Kaggle first if configured
    if kagglehub is not None and config.kaggle_dataset and not config.skip_download:
        try:
            logger.info(f"Downloading Kaggle dataset: {config.kaggle_dataset}...")
            path = kagglehub.dataset_download(config.kaggle_dataset)
            logger.info(f"Dataset available at: {path}")
            copied = _copy_from_path(Path(path))
            if copied > 0:
                logger.info(f"Copied {copied} histology images into {config.input_dir}")
                return
            logger.warning("No usable images found in Kaggle dataset")
        except Exception as e:
            logger.warning(f"Kaggle download failed: {e}")
    
    # Fallback to placeholder images
    logger.info("Using placeholder images (no Kaggle dataset available)")
    sample_urls = [
        'https://picsum.photos/seed/histo1/1024/1024',
        'https://picsum.photos/seed/histo2/1024/1024',
        'https://picsum.photos/seed/histo3/1024/1024'
    ]
    
    for i, url in enumerate(tqdm(sample_urls[:config.max_samples], desc="Downloading samples")):
        dest = config.input_dir / f'sample_{i+1}.png'
        if dest.exists():
            continue
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            dest.write_bytes(response.content)
            logger.debug(f"Downloaded: {dest}")
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")


def preprocess_images(config, logger):
    """Preprocess images to the required size and format."""
    logger.info("Starting image preprocessing...")
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    image_paths = []
    
    for pat in patterns:
        image_paths.extend(config.input_dir.glob(pat))
    
    if not image_paths:
        logger.warning("No images found for preprocessing")
        return []
    
    processed_paths = []
    for img_path in tqdm(image_paths[:config.max_samples], desc="Preprocessing"):
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if img.size != (config.image_size, config.image_size):
                    img = img.resize((config.image_size, config.image_size), Image.BICUBIC)
                
                # Save as PNG for consistency
                target = img_path if img_path.suffix.lower() == '.png' else \
                         config.input_dir / f"{img_path.stem}.png"
                img.save(target)
                processed_paths.append(target)
                
                if target != img_path:
                    try:
                        img_path.unlink(missing_ok=True)
                    except Exception as e:
                        logger.debug(f"Could not remove {img_path}: {e}")
                
                logger.debug(f"Processed: {target}")
                
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
    
    logger.info(f"Preprocessed {len(processed_paths)} images")
    return processed_paths


class PixCellReconstructor:
    """Handles PixCell model loading and image reconstruction."""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        self.pipeline = None
        self.uni_model = None
        self.transform = None
    
    def load_models(self):
        """Load all required models."""
        self.logger.info("Loading models...")
        start_time = time.time()
        
        try:
            # Load VAE
            vae = AutoencoderKL.from_pretrained(
                self.config.vae_name,
                subfolder="vae",
                torch_dtype=self.dtype
            )
            
            # Load PixCell pipeline
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.config.model_name,
                vae=vae,
                custom_pipeline="StonyBrook-CVLab/PixCell-pipeline",
                trust_remote_code=True,
                torch_dtype=self.dtype,
            )
            self.pipeline.to(self.device)
            
            # Load UNI model
            timm_kwargs = {
                'img_size': 224,
                'patch_size': 14,
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5,
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0,
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked,
                'act_layer': torch.nn.SiLU,
                'reg_tokens': 8,
                'dynamic_img_size': True
            }
            
            self.uni_model = timm.create_model(
                self.config.uni_model_name,
                pretrained=True,
                **timm_kwargs
            )
            self.uni_model.eval()
            self.uni_model.to(self.device)
            
            # Create transform
            config = resolve_data_config(self.uni_model.pretrained_cfg, model=self.uni_model)
            self.transform = create_transform(**config)
            
            self.logger.info(f"Models loaded in {time.time() - start_time:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def encode(self, image):
        """Encode image to latent representation."""
        if self.uni_model is None or self.transform is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Rearrange into 16 256x256 patches
        patches = einops.rearrange(
            img_array,
            '(d1 h) (d2 w) c -> (d1 d2) h w c',
            d1=4, d2=4
        )
        
        # Apply transforms to each patch
        patch_tensors = torch.stack([
            self.transform(Image.fromarray(patch)) 
            for patch in patches
        ])
        
        # Get embeddings
        with torch.inference_mode():
            patch_tensors = patch_tensors.to(self.device)
            embeddings = self.uni_model(patch_tensors)
            
        # Reshape to (batch_size, num_patches, embedding_dim)
        return embeddings.unsqueeze(0)
    
    def decode(self, latent):
        """Decode latent representation back to image."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_models() first.")
            
        # Get unconditional embedding for classifier-free guidance
        uncond = self.pipeline.get_unconditional_embedding(latent.shape[0])
        
        # Generate image
        with torch.amp.autocast(device_type=self.device.type):
            result = self.pipeline(
                uni_embeds=latent,
                negative_uni_embeds=uncond,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps
            )
            
        return result.images[0]  # Return first (and only) image


def plot_comparison(original, reconstructed, save_path=None, figsize=(10, 5)):
    """Plot original and reconstructed images side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(reconstructed)
    ax2.set_title('Reconstructed')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.show()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(verbose=args.debug)
    
    # Create config
    config = Config(args)
    
    # Set random seed
    set_seed(config.seed)
    
    # Log configuration
    logger.info("=== Configuration ===")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Input directory: {config.input_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Image size: {config.image_size}")
    logger.info(f"Max samples: {config.max_samples}")
    logger.info(f"Guidance scale: {config.guidance_scale}")
    logger.info(f"Inference steps: {config.num_inference_steps}")
    logger.info(f"Random seed: {config.seed}")
    logger.info("==================")
    
    try:
        # Download and preprocess images
        download_sample_images(config, logger)
        processed_paths = preprocess_images(config, logger)
        
        if not processed_paths:
            logger.error("No images available for processing")
            return
            
        # Initialize reconstructor
        reconstructor = PixCellReconstructor(config, logger)
        if not reconstructor.load_models():
            logger.error("Failed to load models")
            return
        
        # Process each image
        logger.info("Starting image reconstruction...")
        
        for img_path in tqdm(processed_paths, desc="Processing images"):
            try:
                # Load and process image
                with Image.open(img_path) as img:
                    original_img = img.convert('RGB')
                    
                    # Encode and decode
                    latent = reconstructor.encode(original_img)
                    reconstructed_img = reconstructor.decode(latent)
                    
                    # Save results
                    base_name = img_path.stem
                    rec_path = config.output_dir / f'{base_name}_recon.png'
                    comp_path = config.output_dir / f'{base_name}_comparison.png'
                    
                    reconstructed_img.save(rec_path)
                    plot_comparison(original_img, reconstructed_img, comp_path)
                    
                    logger.info(f"Processed: {img_path.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}", exc_info=args.debug)
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=args.debug)
        sys.exit(1)


if __name__ == "__main__":
    main()
