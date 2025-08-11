# PixCell-1024 VAE — Histopathology Synthetic Image Project

Goal
- Use the pretrained PixCell-1024 foundational VAE to generate synthetic histopathology images and perform reconstructions, starting small locally on macOS (MPS) and scaling later if needed.
- Primary notebooks: [00_environment_check.ipynb](notebooks/00_environment_check.ipynb:1), [01_model_inference_pixcell_updated.ipynb](notebooks/01_model_inference_pixcell_updated.ipynb:1), [02_dataset_prep.ipynb](notebooks/02_dataset_prep.ipynb:1).

References
- Model: https://huggingface.co/StonyBrook-CVLab/PixCell-1024
- Paper: https://arxiv.org/html/2506.05127v1

Systemic Overview
- Stage 0 — Decisions & Environment
  - Local (CPU/MPS) for Stage 1–2. Use Conda/mamba. Jupyter for visualization.
- Stage 1 — Minimal Inference
  - Verify environment and generate a few random samples. Save to data/outputs/inference/.
- Stage 2 — Dataset & Reconstructions
  - Use public non-sensitive tiles. Preprocess to 1024 (verify on model card). Encode→decode reconstructions; save comparisons to data/outputs/reconstructions/.
- Stage 3 — Optional Adaptation (later on Colab/GPU)
- Stage 4 — Evaluation (visual + light metrics)
- Stage 5 — Packaging & Sharing

Requirements
- Python 3.10
- Conda/mamba
- PyTorch with MPS (Apple Silicon) — selected dynamically via [torch.device()](notebooks/00_environment_check.ipynb:1).
- Core packages listed in [env/requirements.txt](env/requirements.txt:1) and [env/environment.yml](env/environment.yml:1).

Repository Structure
- notebooks/
  - 00_environment_check.ipynb
  - 01_model_inference_pixcell_updated.ipynb - Main notebook for generating synthetic histopathology images with customizable parameters
  - 02_dataset_prep.ipynb - For processing and reconstructing histopathology images
  - optimized_pixcell_inference.py - Optimized script for batch image generation
  - 04_evaluation_visual_metrics.ipynb (placeholder; add later)
- src/utils/
  - io.py
  - viz.py
  - metrics.py
- data/
  - inputs/
  - outputs/
    - inference/
    - reconstructions/
    - metrics/
- env/
  - requirements.txt
  - environment.yml
- .gitignore
- LICENSE

Quickstart (Conda on macOS)
1) Create and activate the environment:
   conda env create -f [env/environment.yml](env/environment.yml:1)
   conda activate vaes-synth-img
2) Register Jupyter kernel:
   python -m ipykernel install --user --name vaes-synth-img --display-name "vaes-synth-img"
3) Launch notebooks:
   jupyter notebook
4) Open and run:
   - [00_environment_check.ipynb](notebooks/00_environment_check.ipynb:1)
   - [01_model_inference_pixcell.ipynb](notebooks/01_model_inference_pixcell.ipynb:1)
   - [02_dataset_prep.ipynb](notebooks/02_dataset_prep.ipynb:1)

MPS Notes
- The notebooks auto-select mps if [torch.backends.mps.is_available()](notebooks/00_environment_check.ipynb:1) else cpu.
- Expect moderate speed for inference; training is limited locally.

Data & Ethics
- Use only public, non-sensitive tiles for demos.
- Reconstructions assess model fidelity; generation uses random latents.

TODO — Model Loader
- Follow the Hugging Face model card to load PixCell-1024. Stubs in the notebooks raise NotImplementedError. Replace with the official loading/encode/decode API when available. See [01_model_inference_pixcell.ipynb](notebooks/01_model_inference_pixcell.ipynb:1) and [02_dataset_prep.ipynb](notebooks/02_dataset_prep.ipynb:1).

Run Order
- Stage 0: 00 → Stage 1: 01 → Stage 2: 02.

Risks & Performance
- CPU/MPS limits batch size and speed.
- Fine-tuning is recommended on GPU (e.g., Colab) later.

License
- MIT; see [LICENSE](LICENSE:1).