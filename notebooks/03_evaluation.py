import argparse
import torch
from PIL import Image
from pathlib import Path
import lpips
import numpy as np

# Assuming src.utils.metrics and src.utils.io are correctly set up in the PYTHONPATH
from src.utils.metrics import lpips_distance
from src.utils.io import load_image_as_tensor

def main():
    parser = argparse.ArgumentParser(description="Compute LPIPS scores for reconstructed images.")
    parser.add_argument("--original_dir", type=str, required=True,
                        help="Path to the directory containing original images.")
    parser.add_argument("--reconstructed_dir", type=str, required=True,
                        help="Path to the directory containing reconstructed images.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of image pairs to evaluate. If None, evaluate all found pairs.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional path to a file to save individual LPIPS scores.")

    args = parser.parse_args()

    original_dir = Path(args.original_dir)
    reconstructed_dir = Path(args.reconstructed_dir)
    output_file = Path(args.output_file) if args.output_file else None

    if not original_dir.is_dir():
        print(f"Error: Original images directory not found at {original_dir}")
        return
    if not reconstructed_dir.is_dir():
        print(f"Error: Reconstructed images directory not found at {reconstructed_dir}")
        return

    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize LPIPS model
    # The LPIPS model needs to be on the same device as the images
    lpips_model = lpips.LPIPS(net='alex').to(device)

    image_files = sorted(list(original_dir.glob("*.png")) + list(original_dir.glob("*.jpg")))
    if not image_files:
        print(f"No image files found in {original_dir}. Supported formats: .png, .jpg")
        return

    lpips_scores = []
    evaluated_samples = 0

    for original_path in image_files:
        if args.num_samples is not None and evaluated_samples >= args.num_samples:
            break

        reconstructed_path = reconstructed_dir / original_path.name
        if not reconstructed_path.exists():
            print(f"Warning: No reconstructed image found for {original_path.name}. Skipping.")
            continue

        try:
            # Load images and convert to tensors, then move to device
            img_orig = load_image_as_tensor(original_path).to(device)
            img_recon = load_image_as_tensor(reconstructed_path).to(device)

            # Compute LPIPS score
            score = lpips_model(img_orig, img_recon).item()
            lpips_scores.append(score)
            evaluated_samples += 1
            print(f"Processed {original_path.name}: LPIPS = {score:.4f}")

        except Exception as e:
            print(f"Error processing {original_path.name}: {e}")
            continue

    if lpips_scores:
        average_lpips = np.mean(lpips_scores)
        print(f"\n--- Evaluation Summary ---")
        print(f"Total image pairs evaluated: {evaluated_samples}")
        print(f"Average LPIPS score: {average_lpips:.4f}")

        if output_file:
            try:
                with open(output_file, 'w') as f:
                    for score in lpips_scores:
                        f.write(f"{score}\n")
                print(f"Individual LPIPS scores saved to {output_file}")
            except Exception as e:
                print(f"Error saving scores to {output_file}: {e}")
    else:
        print("No LPIPS scores computed. Please check input directories and image formats.")

if __name__ == "__main__":
    main()