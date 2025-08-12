"""Metric stubs. For pilots, use LPIPS for reconstructions; FID needs many samples."""
import lpips
import torch

# Initialize LPIPS model globally to avoid re-loading for each call
_lpips_model = lpips.LPIPS(net='alex')

def lpips_distance(img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
    """
    Computes the LPIPS (Learned Perceptual Image Patch Similarity) distance between two images.

    Args:
        img_a (torch.Tensor): The first image tensor (N, C, H, W), normalized to [-1, 1].
        img_b (torch.Tensor): The second image tensor (N, C, H, W), normalized to [-1, 1].

    Returns:
        torch.Tensor: A scalar tensor representing the LPIPS distance.
    """
    # Ensure the LPIPS model is on the same device as the input tensors
    if img_a.device != _lpips_model.device:
        _lpips_model.to(img_a.device)
    
    # LPIPS expects images normalized to [-1, 1]
    # Assuming input tensors are already normalized as per typical PyTorch image processing.
    # If not, they should be normalized before calling this function.
    
    distance = _lpips_model(img_a, img_b)
    return distance.mean() # Return a scalar value

def fid_placeholder():
    """Document that real FID requires 1k+ images and Inception features."""
    return 'FID not recommended for tiny pilot sample sizes.'