# FixCBM: Concept-Consistent Semi-Supervised Learning for Concept Bottleneck Models via Confidence-Guided Pseudo-Label Propagation

<p align="center">
  <img src="docs/figures/Figure1.jpg" alt="FixCBM framework" width="100%">
</p>

## Overview

FixCBM is a semi-supervised Concept Bottleneck Model framework. It learns from scarce concept annotations by generating pseudo-labels from weakly augmented views, filtering them with per-concept confidence, and enforcing weak-to-strong consistency in concept probability space.

Concept Bottleneck Models improve interpretability by predicting human-understandable concepts before making task predictions, but concept annotations are expensive to collect. FixCBM reduces this annotation burden by using unlabeled concept data through confidence-guided pseudo-label propagation. Instead of relying on spatial alignment or additional heatmap branches, FixCBM applies consistency regularization directly in concept probability space.

## Highlights

- Concept-level consistency learning for semi-supervised CBMs.
- Confidence-guided pseudo-label filtering for unlabeled concept supervision.
- Warmup scheduling to reduce noisy unlabeled losses early in training.
- Dataset loaders and configs for CUB-200-2011, AwA2, PBC, CelebA, and 7-point.

## Repository Structure

```text
FixCBM/
|-- main.py                         # Main entry point for training and evaluation
|-- requirements.txt                # Python dependencies
|-- environment.yml                 # Optional conda environment file
|-- configs/
|   |-- basic_config.py             # Command-line arguments
|   |-- CUB-200-2011.yaml           # CUB-200-2011 configuration
|   |-- 7pt.yaml                    # 7-point skin lesion configuration
|   |-- AwA2.yaml                   # Animals with Attributes 2 configuration
|   |-- PBC.yaml                    # PBC configuration
|   `-- CelebA.yaml                 # CelebA configuration
|-- data/
|   |-- cub_loader.py               # CUB-200-2011 loader and FixMatch augmentations
|   |-- pt_loader.py                # 7-point loader
|   |-- awa2_loader.py              # AwA2 loader
|   |-- pbc_loader.py               # PBC loader
|   `-- celeba_loader.py            # CelebA loader
|-- models/
|   |-- fixcbm.py                   # FixCBM architecture and training steps
|   `-- construction.py             # Model builder and checkpoint loader
|-- train/
|   |-- training.py                 # Training loop, validation, testing, result logging
|   |-- evaluate.py                 # Representation and concept evaluation helpers
|   `-- utils.py                    # Accuracy, backbone wrapping, and Lightning helpers
|-- visualization/
|   `-- heatmap.py                  # Heatmap generation from a trained FixCBM checkpoint
|-- docs/
`-- scripts/
```

## Installation

Create an environment with Python 3.8 or a compatible Python 3.7/3.8 runtime:

```bash
conda create -n fixcbm python=3.8
conda activate fixcbm
pip install -r requirements.txt
```

The current implementation imports external `cem` modules for shared metrics and training utilities. Install or vendor the matching dependency before running training:

```bash
pip install -e /path/to/concept-embedding-models
```

## Dataset Preparation

See [docs/DATASETS.md](docs/DATASETS.md) for expected dataset layouts.

Before running an experiment, edit the corresponding file under `configs/` and set:

- `dataset_config.root_dir`
- `dataset_config.batch_size`
- `dataset_config.num_workers`

## Quick Start

### 1. Install dependencies

```bash
conda create -n fixcbm python=3.8
conda activate fixcbm
pip install -r requirements.txt
```

Install the external CEM dependency used by the shared metrics and helper utilities:

```bash
pip install -e /path/to/concept-embedding-models
```

### 2. Prepare datasets

Download CUB-200-2011 and set `dataset_config.root_dir` in the matching config file:

```text
configs/CUB-200-2011.yaml
```

For example:

```yaml
dataset_config:
  dataset: CUB-200-2011
  root_dir: ./data/CUB_200_2011
```

More dataset layout notes are available in [docs/DATASETS.md](docs/DATASETS.md).

### 3. Run training

Train FixCBM on CUB-200-2011 with 10% concept labels:

```bash
python main.py --dataset CUB-200-2011 --labeled_ratio 0.1 --seed 42
```

An equivalent shell script is provided:

```bash
bash scripts/run_cub.sh
```

### 4. Useful arguments

```text
--dataset         Dataset config name. Use CUB-200-2011 for this example.
--labeled_ratio   Ratio of samples with concept labels.
--seed            Random seed for data split and training.
--image_encoder   Backbone name. Default: resnet34.
--save_path       Output directory. Default: ./checkpoints/.
--device          Training device. Use gpu when CUDA is available.
```

### 5. Outputs

Outputs are written under `checkpoints/` by default. Each run creates a timestamped folder containing:

```text
running.log
args.yml
experiment_config.yaml
results.txt
codes.zip
FixCBM.pt
```

### 6. Heatmap visualization

After training, generate CUB heatmaps from a saved checkpoint directory:

```bash
python visualization/heatmap.py \
  --dataset CUB-200-2011 \
  --checkpoint_dir checkpoints/<run_folder> \
  --model_name FixCBM
```

## Method Overview

FixCBM applies weak-to-strong consistency in concept space:

1. Predict concept probabilities from weakly augmented images.
2. Select high-confidence concept predictions as pseudo-labels.
3. Apply strong augmentation to the same images.
4. Enforce consistency between strong-view predictions and selected pseudo-labels.
5. Use warmup scheduling to reduce early pseudo-label noise.
