#!/bin/bash
# model-lf8: Q3D target model_e_lf8 (8 grounded ports, DC - 60 MHz; data/README.md), default settings.
# Writes runs/model-lf8/{summary.md,summary.json,best/,seeds/}; every seed is also scored as the floating 4-port.
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-100-107}
python -u train.py --target data/model_e_lf8.npz --seeds $SEEDS --deterministic --save-all --out runs/model-lf8 "$@"
