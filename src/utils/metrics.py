"""Metric stubs. For pilots, use LPIPS for reconstructions; FID needs many samples."""
def lpips_distance(img_a, img_b):
    """TODO: Implement LPIPS using the lpips package and torch tensors."""
    raise NotImplementedError('Implement LPIPS distance computation')

def fid_placeholder():
    """Document that real FID requires 1k+ images and Inception features."""
    return 'FID not recommended for tiny pilot sample sizes.'