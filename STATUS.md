# Artifact snapshot status

This directory tracks the manuscript *Branch Evidence and Temporal Context for Selective Event Spotting*. Canonical values live in `results/*.csv` plus `results/summary.json`; `src/render.py` generates every manuscript number from those files. This document intentionally duplicates no metrics.

## Included checks

- Match-grouped validation for selector plus ranker selection.
- One-to-one candidate labels tested against the official SoccerNet convention.
- Tie-aware AURC, risk-at-coverage, Brier score, reliability uncertainty, plus tie bounds.
- Separate offline, past-only, no-agreement, per-half, plus transductive variants.
- Five model seeds for primary tables. League holdout remains an exploratory seed-0 analysis.
- NetVLAD++, CALF, QMF, PAVE, early-fusion, late-fusion, single-branch, plus visual-ensemble baselines.
- Dependency pins, QMF port notes, runtime metadata, plus SHA-256 provenance.

## External requirements

SoccerNet-v2 features, labels, plus action frames remain subject to the SoccerNet access terms. The artifact records acquisition instructions plus Figure 1 source identifiers. Anonymous hosting remains an author-controlled submission step.
