import sys
import torch
import diffusers
import safetensors
import numpy as np

print("=== Environment Check ===")
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
print(f"Diffusers version: {diffusers.__version__}")
print(f"Safetensors version: {safetensors.__version__}")
print("\n=== Important Paths ===")
print(f"Torch path: {torch.__file__}")
print(f"Diffusers path: {diffusers.__file__}")

# Check if we can create a tensor on MPS (Apple Silicon)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    try:
        x = torch.rand(2, 3, device='mps')
        print("\n=== MPS Test ===")
        print(f"Successfully created tensor on MPS: {x}")
        print(f"MPS device name: {torch.mps.get_device_name()}")
    except Exception as e:
        print(f"\n=== MPS Test Failed ===")
        print(f"Error: {str(e)}")
else:
    print("\n=== MPS Not Available ===")
