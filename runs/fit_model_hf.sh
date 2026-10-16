#!/bin/bash
# model-hf: HFSS target model_e_hf (HFSS_4port_V1, 100 kHz - 1 GHz), default settings.
# Writes runs/model-hf/{summary.md,summary.json,best/,seeds/}.
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-100-107}
python -u train.py --target data/model_e_hf.npz --seeds $SEEDS --deterministic --save-all --out runs/model-hf "$@"
