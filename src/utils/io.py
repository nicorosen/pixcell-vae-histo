"""I/O utilities for saving images and ensuring directories exist."""
from pathlib import Path
from PIL import Image

def ensure_dir(path):
    """Create directory path.parents if needed."""
    p = Path(path)
    (p if p.suffix == '' else p.parent).mkdir(parents=True, exist_ok=True)

import torch
import torchvision.transforms as transforms

def save_image(img, path):
    """Save a PIL.Image to path (PNG)."""
    ensure_dir(path)
    img.save(path, format='PNG')

def load_image_as_tensor(path):
    """
    Load an image from path and convert it to a PyTorch tensor.
    The image is resized to 256x256, converted to a tensor, normalized to [-1, 1],
    and has its channel dimension moved to the first position (C, H, W).
    """
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0) # Add batch dimension