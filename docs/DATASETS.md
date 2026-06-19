# Datasets

FixCBM expects dataset paths to be configured in `configs/<dataset>.yaml` under `dataset_config.root_dir`.

## CUB-200-2011

Download the image dataset from the official Caltech CUB-200-2011 page and place/extract it so that `root_dir` points to the `CUB_200_2011` directory.

The CUB loader also expects concept annotation files compatible with the CBM/CEM split format. If your local data follows the original CBM layout, keep the annotation files adjacent to the CUB data directory as expected by `data/cub_loader.py`.

Example:

```yaml
dataset_config:
  dataset: "CUB-200-2011"
  root_dir: ./data/CUB_200_2011
```

## AwA2

Set `dataset_config.root_dir` to your local AwA2 image/attribute directory and verify the selected concept settings in `configs/AwA2.yaml`.

## PBC

Set `dataset_config.root_dir` to the PBC dataset directory. The loader expects image files and metadata used by `data/pbc_loader.py`.

## 7-point

Set `dataset_config.root_dir` to the extracted 7-point dataset directory:

```yaml
dataset_config:
  dataset: "7pt"
  root_dir: ./data/7point
```
