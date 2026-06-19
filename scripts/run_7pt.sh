#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --dataset 7pt \
  --labeled_ratio 0.1 \
  --seed 42
