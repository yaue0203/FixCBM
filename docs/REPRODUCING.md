# Reproducing Experiments

This repository stores experiment settings in YAML files under `configs/`.

## Main Entry Point

```bash
python main.py --dataset CUB-200-2011 --labeled_ratio 0.1 --seed 42
```

Available dataset names:

- `CUB-200-2011`
- `AwA2`
- `PBC`
- `7pt`
- `CelebA`

## Important Arguments

- `--dataset`: selects `configs/<dataset>.yaml` and the matching loader.
- `--labeled_ratio`: proportion of samples with concept annotations.
- `--seed`: controls data splits and training randomness.
- `--image_encoder`: backbone name, default `resnet34`.
- `--save_path`: output directory, default `./checkpoints/`.

## Outputs

Each run creates a timestamped directory under `checkpoints/` containing:

- `running.log`
- `args.yml`
- `experiment_config.yaml`
- `results.txt`
- `codes.zip`
- model checkpoints when `save_model: True`
