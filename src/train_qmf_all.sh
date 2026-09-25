#!/usr/bin/env bash
# Train the real QMF model (5 seeds) and write its predictions for valid/test and the degradation variants.
set -euo pipefail; cd "$(dirname "$0")/.."; PY=.venv/bin/python
for s in 0 1 2 3 4; do
  [[ -f models/qmf_s$s.pt ]] || $PY src/train_branch.py --modality qmf --seed $s --epochs 100 --patience 8 > data/logs/train_qmf_s$s.log 2>&1
  for sp in valid test; do $PY src/predict.py --tag qmf_s$s --split $sp; done
  for p in 0.25 0.5 0.75; do $PY src/predict.py --tag qmf_s$s --split test --visual-dropout $p --out-tag qmf_s${s}_vdrop$p; done
  $PY src/predict.py --tag qmf_s$s --split test --audio-mute 0.5 --out-tag qmf_s${s}_amute0.5
  echo "qmf seed $s done $(date)"
done
echo "train_qmf_all: done"
