"""I/O utilities for saving images and ensuring directories exist."""
from pathlib import Path
from PIL import Image

def ensure_dir(path):
    """Create directory path.parents if needed."""
    p = Path(path)
    (p if p.suffix == '' else p.parent).mkdir(parents=True, exist_ok=True)

def save_image(img, path):
    """Save a PIL.Image to path (PNG)."""
    ensure_dir(path)
    img.save(path, format='PNG')