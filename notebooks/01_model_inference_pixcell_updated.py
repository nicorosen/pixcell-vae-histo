import os
import sys
import random
from pathlib import Path
from datetime import datetime
import warnings
import argparse

# Add the project root to the Python path to allow imports from 'src'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import the PixCellVAELoader from the new wrapper
from src.models.pixcell_vae_wrapper import PixCellVAELoader, ModelLoadingError

# Optional: lower MPS high watermark to release memory more aggressively (harmless on non-MPS)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# Output directory
OUT_DIR = Path("data/outputs/inference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    args = parse_args()

    # Initialize the PixCellVAELoader
    # Get the default config from the wrapper and then apply overrides
    from src.models.pixcell_vae_wrapper import DEFAULT_MODEL_CONFIG
    loader_config = DEFAULT_MODEL_CONFIG.copy() # Start with a copy of the default config

    if args.fast:
        loader_config['generation']['num_inference_steps'] = 12
        loader_config['generation']['guidance_scale'] = 1.0
    if args.steps is not None:
        loader_config['generation']['num_inference_steps'] = int(args.steps)
    if args.guidance is not None:
        loader_config['generation']['guidance_scale'] = float(args.guidance)
    if args.num is not None:
        loader_config['generation']['num_samples'] = int(args.num)
    
    run_seed = int(os.getenv("PIX_SEED", 34)) # Default seed from original script
    if args.seed is not None:
        run_seed = int(args.seed)
    elif args.randomize:
        run_seed = int(datetime.now().timestamp())
    
    pixcell_loader = PixCellVAELoader(model_config=loader_config, base_seed=run_seed, debug=os.getenv("PIX_DEBUG", "0").strip() == "1")
    
    # Echo which token env is set (helps debug auth issues)
    if os.getenv('HUGGING_FACE_HUB_TOKEN'):
        print("Auth: using HUGGING_FACE_HUB_TOKEN from environment/.env")
    elif os.getenv('HF_TOKEN'):
        print("Auth: using HF_TOKEN from environment/.env")

    try:
        # Load models using the wrapper
        print("Initializing models...")
        pixcell_loader.load_models()

        # Set conditioning dir from CLI arg if provided
        cond_dir = args.cond_dir.strip() if args.cond_dir else os.getenv("CONDITION_DIR", "").strip()
        if cond_dir:
            print(f"Conditioning directory set to: {cond_dir}")
            # The wrapper handles loading from cond_dir internally now

        # Generate samples
        print("\nGenerating samples...")
        samples = pixcell_loader.sample(
            num_samples=args.num, # Pass num_samples from args, or let wrapper use its config
            cond_dir=cond_dir
        )

        # Optionally run unconditional control branch
        if args.uncond_test:
            # The wrapper's sample method can handle unconditional generation if cond_dir is empty
            # For a specific "control" image, we might need a dedicated method in the wrapper
            # For now, we'll just generate a regular sample without conditioning if uncond_test is true
            print("Running unconditional control test (generating a sample without specific conditioning).")
            uncond_samples = pixcell_loader.sample(num_samples=1, cond_dir="")
            if uncond_samples:
                uncond_samples[0].save(OUT_DIR / f'control_uncond_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                print(f"Saved uncond-control to {OUT_DIR / 'control_uncond_{}.png'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))}")


        # Display results and save
        if samples:
            print(f"\nSuccessfully generated {len(samples)} samples")
            display_samples(samples)
            for i, img in enumerate(samples):
                img_path = OUT_DIR / f'sample_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{i:02d}.png'
                img.save(img_path)
                print(f"Saved sample {i+1} to {img_path}")
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
        pixcell_loader.clear_memory()
        print("\nDone!")

if __name__ == "__main__":
    main()