#!/usr/bin/env bash
# Wait for feature downloads, then train visual/audio/joint x seeds in three parallel streams.
cd "$(dirname "$0")/.."
while pgrep -f download_features.py > /dev/null; do sleep 30; done
echo "downloads finished: $(find data/SoccerNet -name '*_PCA512.npy' | wc -l) feature files at $(date)"
SEEDS="${SEEDS:-0 1 2 3 4}"
for m in visual audio joint; do
  ( for s in $SEEDS; do
      [[ -f models/${m}_s${s}.pt ]] || .venv/bin/python src/train_branch.py --modality $m --seed $s --epochs 100 --patience 8 > data/logs/train_${m}_s${s}.log 2>&1
      echo "trained ${m}_s${s}: $(tail -1 data/logs/train_${m}_s${s}.log)"
    done ) &
done
wait
echo "ALL TRAINING DONE at $(date)"
