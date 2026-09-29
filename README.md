# Supplementary Artifact — Branch Evidence and Temporal Context for Selective Event Spotting

**Paper:** Branch Evidence and Temporal Context for Selective Event Spotting  
*Venue:** ACIIDS 2027 (Springer LNCS/LNAI)  
**Authors:** Tien-Anh Nguyen, Thanh-Hai Tran, Xuan-Bach Le — Ho Chi Minh City University of Technology (HCMUT), VNUHCM

This repository contains the source code, canonical results, and reproduction instructions for the paper. Every number reported in the paper is generated from `results/*.csv` and `results/summary.json` by `src/render.py`; no value is hard-coded in the LaTeX source.

---

## Repository structure

```
artifact/
├── src/                    # full pipeline
│   ├── train_branch.py     # visual / audio / joint / QMF branch training
│   ├── predict.py          # inference (clean + corruption variants)
│   ├── run_pipeline.py     # five-seed selector + ranker fitting and evaluation
│   ├── analysis.py         # candidate construction, features, metrics, reranking
│   ├── render.py           # generates all paper tables, macros, and Figure 2
│   ├── merge_results.py    # merges per-seed outputs into canonical CSVs
│   └── common.py           # shared utilities (paths, SoccerNet loading, NMS)
├── baselines/
│   ├── CALF/               # official CALF code (retrained on our features)
│   ├── TemporallyAwarePooling/  # official NetVLAD++ code (retrained)
│   └── patches/
│       └── calf_multiseed.py   # adds --seed argument to vanilla CALF
├── tests/
│   └── test_analysis.py    # 10 targeted unit tests
├── results/                # canonical result files (CSV / JSON)
├── provenance/
│   ├── manifest.sha256     # SHA-256 hashes of source and output files
│   └── run.json            # runtime environment and code revision
├── run_all.sh              # end-to-end stage driver
├── requirements-lock.txt   # audited Python dependency versions
├── QMF_PORT.md             # pinned upstream QMF revision and task-specific changes
├── REPRODUCE.md            # step-by-step reproduction instructions
└── CITATION.cff            # citation metadata
```

---

## Method summary

**AGREE** is a post-hoc reliability layer over fixed audio-visual spotting branches. It does not retrain the backbone.

1. **Candidate generation.** Per-class NMS on the fused score `P_f = ω P_v + (1−ω) P_a` produces candidates above a validation-selected threshold τ₀.
2. **Feature extraction.** Each candidate carries a 42-dimensional representation: 6 local (branch scores + 3 agreement diagnostics), 5 branch context (±5 s), 12 event context (15/30 s windows), 19 clock/history features.
3. **Selector head `g_s`.** Logistic regression with class-specific interactions, fitted on match-grouped validation folds. Pooled isotonic calibration yields the reliability score `r_s(e)`. Candidates with `r_s(e) < θ` are withheld.
4. **Ranker head `g_r`.** A separate logistic head, selected by out-of-fold validation mAP, reorders candidates while preserving the fused-score multiset within each (match, half, class) group.

Candidate correctness is defined by one-to-one same-class matching within ±5 s. This label is distinct from the SoccerNet loose average-mAP, which averages tolerances from 5 to 60 s.

---

## Canonical result files

| File | Contents |
|---|---|
| `results/main_results.csv` | Spotting performance (loose / tight mAP) for all systems and seeds |
| `results/selective.csv` | AURC, Risk@80/90%, tie bounds, ECE, Brier score |
| `results/ablation.csv` | Feature, causal, no-agreement, and reranking ablations |
| `results/degradation.csv` | Four test-time corruption conditions across five seeds |
| `results/diagnostic.csv` | Branch agreement and Spearman correlation diagnostics |
| `results/rare.csv` | Rare-class calibration evidence |
| `results/league.csv` | Seed-0 exploratory league holdouts |
| `results/summary.json` | Selected thresholds, hyperparameters, bootstrap settings |
| `results/baselines_official.json` | NetVLAD++ and CALF per-run scores |
| `provenance/manifest.sha256` | SHA-256 hashes for source and output files |

> `results/preds/` (~5.4 GB) and trained checkpoints (~144 MB) are hosted on Hugging Face at `lexuanbach/sn-agree` due to file-size limits.

---

## Licensed inputs

This repository does **not** redistribute SoccerNet data or third-party audio features.

- **SoccerNet-v2** labels and ResNet-152 PCA-512 visual features: obtain via the official SoccerNet package (`src/download_features.py`).
- **VGGish audio features**: public release of Vanderplaetse & Dupont (2020); align with `src/prepare_audio.py`.

Expected paths after acquisition:
```
artifact/data/SoccerNet/          # labels + visual features
artifact/data/audio/features/     # aligned VGGish arrays
artifact/results/preds/           # predictions (from Hugging Face or full training)
```

---

## Setup

```bash
cd artifact
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
make test          # 10 unit tests should pass
```

`requirements-lock.txt` records the audited dependency versions (Python 3.14.6, PyTorch 2.12.0, scikit-learn 1.9.0, SoccerNet 0.1.62). Platform-specific PyTorch substitutions should be recorded in `provenance/run.json`.

---

## Reproducing the paper tables

### Option A — from released predictions (minutes)

Place the prediction tree from `lexuanbach/sn-agree` under `results/preds/`, then:

```bash
PARALLEL=1 ./run_all.sh analyze
cd ../manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
../artifact/.venv/bin/python ../artifact/src/write_provenance.py
```

This refits the lightweight selector and ranker heads, regenerates all CSV files, JSON summaries, LaTeX tables, figures, and macros, then builds the PDF.

### Option B — full reproduction from scratch (compute-intensive)

```bash
./run_all.sh all
cd ../manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
../artifact/.venv/bin/python ../artifact/src/write_provenance.py --full
```

Trains visual, audio, joint, and QMF models for seeds 0–4, runs all predictions and corruption variants, then performs the analysis and renders the paper. Existing checkpoints are reused.

See `REPRODUCE.md` for detailed instructions and expected intermediate outputs.

---

## Evaluation notes

- Validation splits are grouped by match (five-fold `GroupKFold`).
- Candidate correctness uses one-to-one same-class greedy matching within ±5 s.
- AURC uses the expected risk under a uniform ordering within each tied-score block; best/worst tie bounds are stored in `selective.csv`.
- Seed-0 confidence intervals use 1,000 paired match bootstrap resamples.
- The full AGREE model is **offline**: it reads up to 30 s of future context. The 35-feature past-only variant is the causal control.
- The 75% visual-dropout condition shows that severe modality failure is the operating boundary; it is retained as a documented scope statement.
- The QMF baseline follows the published training objective; deviations from the upstream code are documented in `QMF_PORT.md`.

---

## Citation

If you use this code or results, please cite:
```
Nguyen, T.-A., Tran, T.-H., Le, X.-B. (2027).
Branch Evidence and Temporal Context for Selective Event Spotting.
In: Proceedings of ACIIDS 2027. Springer LNCS/LNAI.
```
See `CITATION.cff` for machine-readable citation metadata.
