#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for L in england_epl europe_uefa-champions-league france_ligue-1 germany_bundesliga italy_serie-a spain_laliga; do
  ( for m in visual audio; do
      [[ -f models/${m}_s0_xl-${L}.pt ]] || .venv/bin/python src/train_branch.py --modality $m --seed 0 --epochs 100 --patience 8 --exclude-league $L > data/logs/train_${m}_xl-${L}.log 2>&1
      for sp in valid test; do .venv/bin/python src/predict.py --tag ${m}_s0_xl-${L} --split $sp >> data/logs/train_${m}_xl-${L}.log 2>&1; done
      echo "league model done ${m} ${L}"
    done ) &
  # two leagues at a time
  if [[ "$L" == "europe_uefa-champions-league" || "$L" == "germany_bundesliga" ]]; then wait; fi
done
wait; echo "ALL LEAGUE DONE at $(date)"
