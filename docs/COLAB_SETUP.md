# Google Colab Setup for PixCell-VAE-Histo

This document provides instructions for setting up and running the PixCell-VAE-Histo project on Google Colab, leveraging its GPU resources for faster model inference and potential future fine-tuning.

## 1. Open a New Colab Notebook

Go to [Google Colab](https://colab.research.google.com/) and create a new notebook (`File > New notebook`).

## 2. Change Runtime Type to GPU

To utilize Colab's GPU, change the runtime type:
- Click on `Runtime` in the top menu.
- Select `Change runtime type`.
- In the "Hardware accelerator" dropdown, choose `GPU`.
- Click `Save`.

## 3. Mount Google Drive (Optional, for persistent data/models)

If you want to store your input data, generated outputs, or model weights persistently across Colab sessions, you can mount your Google Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Your Google Drive will be accessible at `/content/drive/MyDrive/`. You can then create a project folder there (e.g., `/content/drive/MyDrive/pixcell-vae-histo`) and copy your project files into it.

## 4. Clone the Repository

If your project is hosted on GitHub (or another Git provider), clone it into your Colab environment. Replace `YOUR_REPO_URL` with the actual URL of your repository.

```bash
!git clone YOUR_REPO_URL
%cd pixcell-vae-histo # Change to your project directory name
```

If you mounted Google Drive, you might want to clone it into your Drive:
```bash
%cd /content/drive/MyDrive/
!git clone YOUR_REPO_URL
%cd YOUR_PROJECT_FOLDER_NAME # e.g., pixcell-vae-histo
```

## 5. Set up the Conda Environment (or install dependencies directly)

Colab environments are typically based on Ubuntu and come with `pip` and `conda` (or `mamba`) pre-installed. You can either recreate your Conda environment or install dependencies directly via `pip`.

### Option A: Recreate Conda Environment (Recommended for consistency)

This option ensures your Colab environment closely matches your local Conda setup.

```bash
# Install Mamba (faster than Conda for environment creation)
!conda install mamba -n base -c conda-forge -y

# Create the environment using your environment.yml
# Adjust the path if your env/environment.yml is not at the project root
!mamba env create -f env/environment.yml

# Activate the environment (this might require restarting the runtime or running cells separately)
# Note: Colab often handles environment activation implicitly after creation for subsequent cells.
# If not, you might need to restart runtime and select the new kernel.
```

### Option B: Install Dependencies via Pip (Simpler, but less strict environment control)

If you prefer a simpler setup and don't need strict Conda environment replication, you can install directly from `requirements.txt`.

```bash
# Install dependencies from requirements.txt
# Adjust the path if your env/requirements.txt is not at the project root
!pip install -r env/requirements.txt
```

## 6. Set Hugging Face Token

The PixCell model requires authentication with Hugging Face. Set your Hugging Face token as an environment variable. **Do NOT hardcode your token directly in the notebook.**

```python
import os
from getpass import getpass

# Get your Hugging Face token securely
hf_token = getpass("Enter your Hugging Face token: ")
os.environ["HF_TOKEN"] = hf_token
# Or if you prefer HUGGING_FACE_HUB_TOKEN
# os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
```

## 7. Run Your Scripts/Notebooks

Now you can run your Python scripts or Jupyter notebooks.

### Example: Running 01_model_inference_pixcell_updated.py

```python
# Ensure you are in the correct directory (e.g., pixcell-vae-histo)
# If you cloned into /content/drive/MyDrive/pixcell-vae-histo, you might need:
# %cd /content/drive/MyDrive/pixcell-vae-histo

!python notebooks/01_model_inference_pixcell_updated.py --num 4 --cond_dir data/inputs/colon_imgs
```

### Example: Running 02_dataset_preprocess.py

```python
!python notebooks/02_dataset_preprocess.py --max-samples 5
```

### Example: Running 03_evaluation.py

```python
!python notebooks/03_evaluation.py --original_dir data/inputs --reconstructed_dir data/outputs/reconstructions --max-samples 5
```

## Troubleshooting

*   **`ModuleNotFoundError: No module named 'src'`**: Ensure you have added the project root to `sys.path` in your scripts or are running them from the correct working directory. The `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` line in your scripts should handle this if they are run from `notebooks/`.
*   **Memory Issues (OOM)**: If you encounter "CUDA out of memory" or similar errors, try:
    *   Reducing `num_samples` or batch sizes in your scripts.
    *   Lowering `num_inference_steps`.
    *   Using `torch.float16` (if not already) for models on GPU.
    *   Calling `torch.cuda.empty_cache()` (or `torch.mps.empty_cache()` for MPS) periodically.
*   **Hugging Face Authentication**: Double-check your `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` environment variable. Ensure it has the correct permissions (read access to the model).