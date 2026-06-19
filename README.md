# FixCBM

Code release for **Concept-Consistent Semi-Supervised Learning for Concept Bottleneck Models via Confidence-Guided Pseudo-Label Propagation**.

FixCBM is a semi-supervised Concept Bottleneck Model framework. It learns from scarce concept annotations by generating pseudo-labels from weakly augmented views, filtering them with per-concept confidence, and enforcing weak-to-strong consistency in concept probability space.

<p align="center">
  <img src="docs/figures/Figure1.jpg" alt="FixCBM framework" width="100%">
</p>

## Highlights

- Concept-level consistency learning for semi-supervised CBMs.
- Confidence-guided pseudo-label filtering for unlabeled concept supervision.
- Warmup scheduling to reduce noisy unlabeled losses early in training.
- Dataset loaders and configs for CUB-200-2011, AwA2, PBC, CelebA, and 7-point.

## Repository Layout

```text
FixCBM/
  configs/        Dataset and experiment configs
  data/           Dataset loaders and semi-supervised split logic
  docs/           Dataset, reproduction, and project-structure notes
  models/         FixCBM model and model construction code
  scripts/        Example run scripts
  train/          Training and evaluation loops
  visualization/  Heatmap visualization script
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

Run CUB-200-2011 with 10% concept labels:

```bash
python main.py --dataset CUB-200-2011 --labeled_ratio 0.1 --seed 42
```

Run 7-point with the default config:

```bash
python main.py --dataset 7pt --labeled_ratio 0.1 --seed 42
```

Outputs are written under `checkpoints/` by default. Each run stores logs, resolved arguments, resolved experiment config, results, and a zip snapshot of Python source files.

## Method

FixCBM trains on labeled samples with supervised concept and task losses. For unlabeled concept annotations, it:

1. Predicts per-concept probabilities from weakly augmented views.
2. Converts confident predictions into pseudo-labels.
3. Masks uncertain concept predictions.
4. Enforces consistency on strongly augmented views.
5. Applies warmup so unlabeled concept loss increases gradually.
