# Project Structure

## Root Files

- `main.py`: training entry point for FixCBM experiments.
- `requirements.txt`: Python dependencies for training, evaluation, and heatmap visualization.
- `environment.yml`: optional Conda environment wrapper around `requirements.txt`.
- `README.md`: repository overview and quick-start instructions.
- `LICENSE`: repository license.

## Directories

- `configs/`: command-line argument definitions and dataset-specific YAML configs. Each YAML file contains a single `FixCBM` run.
- `data/`: dataset loaders for CUB-200-2011, AwA2, PBC, CelebA, and 7-point.
- `docs/`: dataset notes, reproduction notes, project structure, and paper figures used by the README.
- `models/`: `fixcbm.py` contains the FixCBM model implementation, and `construction.py` builds or loads FixCBM models.
- `scripts/`: short shell scripts for common CUB and 7-point runs.
- `train/`: training loop, evaluation helpers, result aggregation, and shared training utilities.
- `visualization/`: heatmap generation script for a trained FixCBM checkpoint.

## Main Flow

```text
main.py
  -> configs/<dataset>.yaml
  -> data/*_loader.py
  -> models/construction.py
  -> models/fixcbm.py
  -> train/training.py and train/evaluate.py
```
