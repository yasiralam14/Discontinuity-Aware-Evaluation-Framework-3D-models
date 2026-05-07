# Quantifying Novel View Synthesis at Sharp Boundaries: A Discontinuity-Aware Evaluation Framework

> **Master's Thesis** — Stevens Institute of Technology
> **Advisor:** Philippos Mordohai
> **Author:** Yasir Alam  
> **Contact:** yasir.alam14@gmail.com

---

## Overview

Standard Novel View Synthesis (NVS) benchmarks compute photometric metrics (PSNR, SSIM, LPIPS) as pixel-wise averages over entire images. This treats low-frequency smooth regions and high-frequency sharp boundary regions identically, masking systematic reconstruction failures at object edges and depth discontinuities — precisely where non-Gaussian 3D primitives are expected to excel or fail.

This repository contains the complete research code for the thesis:

**QUANTIFYING NOVEL VIEW SYNTHESIS AT SHARP BOUNDARIES: A DISCONTINUITY-AWARE EVALUATION FRAMEWORK**

The framework introduces three key contributions:

1. **Discontinuity-Aware Evaluation Framework** — A custom evaluation pipeline using Laplacian pyramid zoning to isolate and measure localized photometric errors at spatial discontinuities.
2. **Comprehensive Benchmarking** — Quantitative evaluation of diverse non-Gaussian 3D primitives across multiple scene datasets.
3. **Targeted Spatial Loss** — A training-time spatial loss modification that amplifies RGB and SSIM loss at sharp discontinuities.

---

## Repository Structure

```
thesis/
├── automation_scripts/          # End-to-end training + rendering + evaluation pipelines
│   ├── train_*.sh               # Training scripts per model (3DCS, QGS, Beta-Splatting, etc.)
│   ├── run_*.sh                 # Rendering scripts per model
│   └── run_all.sh               # Full pipeline runner
│
├── metrics/                     # Discontinuity-aware evaluation framework (core contribution)
│   ├── metrics.py               # Single-zone masked evaluation (zone 4 = sharpest)
│   ├── multi_metrics.py         # Multi-zone evaluation (zones 1–4 from Laplacian pyramid)
│   ├── hyper_multi_metrics.py   # Variant for flat directory structure
│   ├── nvs/
│   │   └── nvs_metrics.py       # Masked PSNR, SSIM, LPIPS with spatial mask support
│   ├── image_utils.py           # PSNR/SSIM utilities
│   ├── lpipsPyTorch/            # LPIPS implementation (with mask support)
│   ├── compile_results_to_csv.py
│   ├── multi_compile_results_to_csv.py
│   ├── consolodite_view_metrics.py
│   ├── multi_consolodite_view_metrics.py
│   ├── generate_triplets.py     # Side-by-side GT / Render / Mask visualization
│   ├── count_mask_fraction.py
│   ├── save_max_images.py
│   ├── group_results.py
│   └── run_metrics.sh           # Batch metric runner over all model outputs
│
├── models/                      # 3DGS-family model codebases (submodule-style)
│   ├── gaussian-splatting/      # 3D Gaussian Splatting (baseline + spatial loss variant)
│   │   ├── train.py             # Standard 3DGS training
│   │   ├── train_masked.py      # Spatial-loss variant (Contribution 3)
│   │   ├── utils/
│   │   │   ├── loss_utils.py    # masked_l1_loss, masked_dssim implementations
│   │   │   └── image_utils.py   # laplacian_pyramid_th function
│   │   └── ...
│   ├── 2dgs/                    # 2D Gaussian Splatting
│   ├── 3dcs/                    # 3D Convex Splatting
│   ├── 3dhgs/                   # 3D Hierarchical Gaussian Splatting
│   ├── QGS/                     # Quadric Gaussian Splatting
│   ├── beta_splatting/          # Beta distribution Splatting
│   ├── linprim/                 # Linear Primitives
│   ├── rade_gs/                 # RaDe-GS (Rasterizing Depth in Gaussian Splatting)
│   ├── radfoam/                 # Radiance Foam
│   └── triangle_splatting2/     # Triangle Splatting
│
└── renders/                     # Mask generation scripts and visualization tools
    ├── run_laplacian_and_save_masks.py  # Core mask generation tool
    ├── multi_mask_creation.py
    ├── visualize_mask.py
    └── make_tables/             # Result table generation scripts
```

---

## Contributions

### 1. Discontinuity-Aware Evaluation Framework

The core idea is to use a **Laplacian Pyramid** to segment each ground-truth image into 4 spatial zones based on local frequency content, then compute PSNR/SSIM/LPIPS independently within each zone.

**Zone assignment (1 = smoothest → 4 = sharpest discontinuity):**

```
Zone 1  →  Coarse-scale changes (very low frequency)
Zone 2  →  Mid-scale changes
Zone 3  →  Fine-scale changes
Zone 4  →  Pixel-level sharp boundaries / depth discontinuities
```

**Laplacian Pyramid Zoning (`image_utils.py`):**
```python
def laplacian_pyramid_th(image: Tensor, tau: float) -> Tensor:
    """
    Constructs a 4-level Laplacian pyramid and assigns each pixel
    to a zone (1–4) based on the finest pyramid level at which
    the inter-scale absolute difference exceeds threshold tau.
    """
```

**Mask generation for evaluation:**
```bash
python renders/run_laplacian_and_save_masks.py \
    --input_dir /path/to/gt_images \
    --output_dir /path/to/masks \
    --tau 0.05 \
    --device cuda
```

**Running masked evaluation:**
```bash
# Single-zone evaluation (sharpest zone only)
python metrics/metrics.py -m /path/to/model/output/*

# Multi-zone evaluation (all 4 zones)
python metrics/multi_metrics.py -m /path/to/model/output/*

# Batch evaluation over all models
cd metrics/
bash run_metrics.sh
```

**Output format (`multi_all_scenes_results.json`):**
```json
{
  "scene_name": {
    "SSIM": 0.82, "PSNR": 28.5, "LPIPS": 0.12,
    "Masked1 SSIM": 0.78, "Masked1 PSNR": 26.1, "Masked1 LPIPS": 0.15,
    "Masked2 SSIM": 0.75, "Masked2 PSNR": 24.8, "Masked2 LPIPS": 0.18,
    "Masked3 SSIM": 0.71, "Masked3 PSNR": 22.3, "Masked3 LPIPS": 0.23,
    "Masked4 SSIM": 0.66, "Masked4 PSNR": 20.1, "Masked4 LPIPS": 0.31
  },
  "average": { ... }
}
```

---

### 2. Benchmarking of Non-Gaussian Primitives

We evaluate the following models under the discontinuity-aware framework:

| Model | Primitive Type | Notes |
|---|---|---|
| 3D Gaussian Splatting | Gaussian ellipsoids | Baseline |
| 2D Gaussian Splatting | 2D Gaussian disks | Improved surface reconstruction |
| 3D Convex Splatting (3DCS) | Convex polytopes | Non-Gaussian geometry |
| Quadric Gaussian Splatting (QGS) | Quadric surfaces | Higher-order geometry |
| Beta Splatting | Beta distribution primitives | Bounded support |
| Linear Primitives (LinPrim) | Linear basis functions | |
| RaDe-GS | Gaussians + depth | Depth-regularized |
| Radfoam | Foam / Voronoi cells | Non-parametric |
| Triangle Splatting | Triangle meshes | Rasterization-based |

**Datasets used:**
- **Mip-NeRF 360** (indoor + outdoor unbounded scenes)
- **Tanks and Temples (TNT)** (large-scale outdoor scenes)
- **Deep Blending** (indoor scenes)

---

### 3. Targeted Spatial Loss

The spatial loss modification amplifies both the L1 and D-SSIM loss specifically at pixels belonging to the two sharpest Laplacian pyramid zones (zones 3 and 4).

**Implementation (`models/gaussian-splatting/train_masked.py`):**
```python
# Compute Laplacian zone mask from ground-truth image
mask_full = laplacian_pyramid_th(gt_image.unsqueeze(0), tau=0.05).squeeze(0)
mask3 = (mask_full == 3).float().cuda()
mask4 = (mask_full == 4).float().cuda()

# Apply spatially-weighted loss
Ll1 = masked_l1_loss(image, gt_image, mask3, mask4, loss_weight=loss_weight)
dssim_value = masked_dssim(image.unsqueeze(0), gt_image.unsqueeze(0),
                           mask3, mask4, loss_weight=loss_weight)

loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * dssim_value
```

**Masked L1 loss (`utils/loss_utils.py`):**
```python
def masked_l1_loss(network_output, gt, mask1, mask2, loss_weight):
    l1_diff = torch.abs(network_output - gt)
    weight_mask = torch.where((mask1 > 0) | (mask2 > 0), loss_weight, 1.0)
    return (l1_diff * weight_mask).mean()
```

**Training with spatial loss:**
```bash
python models/gaussian-splatting/train_masked.py \
    -s /path/to/scene \
    -m /path/to/output \
    --loss_weight 2.0 \
    --iterations 30000
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU
- `micromamba` or `conda` for environment management

### Environment Setup

Each model has its own conda environment. Install environments per model:

```bash
# Example: Gaussian Splatting
cd models/gaussian-splatting
conda env create -f environment.yml
conda activate gaussian_splatting

# Example: 3D Convex Splatting
cd models/3dcs/convex-splatting
micromamba env create -f environment.yml
micromamba activate convex_splatting
```

### Metrics Environment

```bash
pip install torch torchvision tqdm Pillow numpy
```

---

## Full Pipeline

### Step 1: Generate Laplacian Masks

```bash
python renders/run_laplacian_and_save_masks.py \
    --input_dir /path/to/dataset/gt \
    --output_dir /path/to/masks \
    --tau 0.05 \
    --device cuda
```

### Step 2: Train Models

Use the automation scripts for each model:

```bash
# Train all scenes for a specific model
cd automation_scripts/
bash train_3dcs.sh       # 3D Convex Splatting
bash train_betasplatting.sh
bash train_QGS.sh
# etc.
```

### Step 3: Render Test Views

```bash
bash automation_scripts/run_3dcs.sh
bash automation_scripts/run_betasplatting.sh
# etc.
```

### Step 4: Evaluate with Discontinuity-Aware Metrics

```bash
cd metrics/
bash run_metrics.sh   # Runs multi_metrics.py over all model output dirs
```

### Step 5: Compile Results

```bash
python metrics/multi_compile_results_to_csv.py
```

---

## Masked LPIPS

The LPIPS implementation in `metrics/nvs/lpipsPyTorch` has been extended to support spatial masks, enabling zone-specific perceptual similarity measurement. Masked LPIPS computes the perceptual distance only over pixels within the specified zone mask.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@mastersthesis{alam2025discontinuity,
  title     = {Quantifying Novel View Synthesis at Sharp Boundaries:
               A Discontinuity-Aware Evaluation Framework},
  author    = {Alam, Yasir},
  school    = {University of Li{\`e}ge},
  year      = {2025},
  note      = {TELIM Research Group}
}
```

---

## License

The original 3D Gaussian Splatting code is:
- Copyright (C) 2023, Inria / GRAPHDECO research group
- Licensed for non-commercial, research and evaluation use under `LICENSE_GS.md`

Modifications and new contributions are:
- Copyright (C) 2025, University of Liège / TELIM research group
- Licensed under `LICENSE.md`

For inquiries: jan.held@uliege.be

---

## Acknowledgements

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) (Kerbl et al., 2023)
- [2D Gaussian Splatting](https://surfsplatting.github.io/)
- [RaDe-GS](https://github.com/BaowenZ/RaDe-GS)
- [Radfoam](https://github.com/theialab/radfoam)
- GRAPHDECO Research Group, Inria
- TELIM Research Group, University of Liège
