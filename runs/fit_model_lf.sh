#!/bin/bash
# model-lf: Q3D target model_e_lf (4 floating ports, DC - 60 MHz; data/README.md), default settings.
# Writes runs/model-lf/{summary.md,summary.json,best/,seeds/}.
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-100-107}
python -u train.py --target data/model_e_lf.npz --seeds $SEEDS --deterministic --save-all --out runs/model-lf "$@"
