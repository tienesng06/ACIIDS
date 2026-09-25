# Reproducing the AGREE results

## Scope

The artifact has two paths:

- **Table reproduction** reuses supplied predictions and rebuilds all CSV, JSON, TeX tables, plots, and macros.
- **Full reproduction** additionally trains every visual, audio, joint, and QMF model and creates their predictions.

SoccerNet-v2 labels plus ResNet-152 PCA-512 features are subject to research-use terms. The same restriction applies to the VGGish audio features of Vanderplaetse plus Dupont. These inputs are not redistributed.

## Environment

The audited environment used Python 3.14.6. From `artifact/`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
make test
```

If the pinned PyTorch wheel is unavailable for a platform, install the corresponding official PyTorch build first, then install the remaining pins without dependency resolution and record the substitution in `provenance/run.json`.

## Inputs

1. Run `.venv/bin/python src/download_features.py` to obtain the SoccerNet-v2 labels and PCA-512 visual features under `data/SoccerNet/`.
2. Follow `src/prepare_audio.py` to place aligned VGGish arrays under `data/audio/features/`.
3. For table-only reproduction, place the supplied prediction tree under `results/preds/`. Its top-level tags plus file hashes belong in the anonymous release manifest.

The standard split comes from the installed SoccerNet package. The analysis uses 300 training, 100 validation, and 100 test matches where the required licensed inputs are present.

## Table-only reproduction

```bash
./run_all.sh analyze
cd ../manuscript
latexmk -pdf main.tex
../artifact/.venv/bin/python ../artifact/src/write_provenance.py
```

This recomputes the selector/ranker with five match-grouped validation folds, uses one-to-one candidate labels within ±5 seconds, evaluates tie-aware AURC, writes `results/*.csv` and `results/summary.json`, regenerates `manuscript/generated/`, builds the PDF, and then writes a compact SHA-256 provenance manifest.

## Full reproduction

```bash
./run_all.sh all
cd ../manuscript
latexmk -pdf main.tex
../artifact/.venv/bin/python ../artifact/src/write_provenance.py --full
```

`run_all.sh all` trains visual, audio, joint, plus QMF models for seeds 0–4. It predicts validation/test plus corruption variants. It runs optional leave-one-league-out models. It performs the analysis, then renders the paper assets. Training is compute-intensive. Existing checkpoints are reused. Remove an individual checkpoint only when that specific model must be retrained.

`QMF_PORT.md` pins the audited upstream revision. It lists every change needed for framewise SoccerNet spotting.

## Expected checks

- `make test` passes matching and tied-risk unit tests.
- `results/summary.json` records `validation_folds` and `aurc_ties`.
- Manuscript numbers are generated only by `src/render.py`.
- `provenance/manifest.sha256` hashes source, compact results, generated TeX, and `manuscript/main.pdf`.
- `provenance/run.json` records the code revision and runtime environment.

## Candidate and metric definitions

A candidate is a per-class NMS output above the validation-selected operating threshold. Correctness is assigned one-to-one to a same-class ground-truth event within ±5 seconds. This candidate-conditional label differs from the loose SoccerNet average-mAP family, which averages tolerances from 5 to 60 seconds. AURC uses the expected within-block order for equal reliability scores, preventing isotonic ties from inheriting file or row order.
