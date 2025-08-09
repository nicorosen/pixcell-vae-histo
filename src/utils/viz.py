"""Visualization utilities: image grids, etc."""
from PIL import Image

def make_image_grid(images, rows, cols):
    """Make a simple grid from a list of PIL images. Returns a PIL.Image."""
    assert len(images) == rows * cols, 'rows * cols must equal len(images)'
    w, h = images[0].size
    grid = Image.new('RGB', (cols * w, rows * h))
    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(im, (c * w, r * h))
    return grid