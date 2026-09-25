#!/usr/bin/env bash
# SN-Agree artifact: end-to-end run. Assumes data/SoccerNet (labels + PCA512 features) and data/audio/features exist.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
SEEDS="${SEEDS:-0 1 2 3 4}"
STAGE="${1:-all}"
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  for m in visual audio joint; do for s in $SEEDS; do
    [[ -f models/${m}_s${s}.pt ]] || $PY src/train_branch.py --modality $m --seed $s
  done; done
  for s in $SEEDS; do
    [[ -f models/qmf_s${s}.pt ]] || $PY src/train_branch.py --modality qmf --seed $s --epochs 100 --patience 8
  done
fi
if [[ "$STAGE" == "all" || "$STAGE" == "predict" ]]; then
  for m in visual audio joint qmf; do for s in $SEEDS; do for sp in valid test; do
    $PY src/predict.py --tag ${m}_s${s} --split $sp
  done; done; done
  for s in $SEEDS; do
    for p in 0.25 0.5 0.75; do
      $PY src/predict.py --tag visual_s${s} --split test --visual-dropout $p --out-tag visual_s${s}_vdrop${p}
      $PY src/predict.py --tag qmf_s${s} --split test --visual-dropout $p --out-tag qmf_s${s}_vdrop${p}
    done
    $PY src/predict.py --tag audio_s${s} --split test --audio-mute 0.5 --out-tag audio_s${s}_amute0.5
    $PY src/predict.py --tag qmf_s${s} --split test --audio-mute 0.5 --out-tag qmf_s${s}_amute0.5
  done
fi
if [[ "$STAGE" == "all" || "$STAGE" == "league" ]]; then
  for L in england_epl europe_uefa-champions-league france_ligue-1 germany_bundesliga italy_serie-a spain_laliga; do for m in visual audio; do
    [[ -f models/${m}_s0_xl-${L}.pt ]] || $PY src/train_branch.py --modality $m --seed 0 --exclude-league $L
    for sp in valid test; do $PY src/predict.py --tag ${m}_s0_xl-${L} --split $sp; done
  done; done
fi
if [[ "$STAGE" == "all" || "$STAGE" == "analyze" ]]; then
  if [[ "${PARALLEL:-0}" == "1" ]]; then   # one process per seed (same outputs as the sequential run), then merge
    for s in $SEEDS; do $PY src/run_pipeline.py --seeds $s --ens-seeds $SEEDS --out-tag _s$s > data/logs/pipeline_s$s.log 2>&1 & done; wait
    $PY src/merge_results.py --seeds $SEEDS
  else
    $PY src/run_pipeline.py --seeds $SEEDS
  fi
  $PY src/render.py
fi
echo "run_all.sh: stage '$STAGE' complete"
