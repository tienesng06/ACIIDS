# QMF port audit

The baseline is a task port of Quality-aware Multimodal Fusion. It is not an exact reproduction of the source paper's image-text benchmarks.

## Pinned reference

- upstream repository: `https://github.com/QingyangZhang/QMF`
- audited commit: `fe6c4c6ef7cb23f0a89594ee413d485f1854268b`
- reference files: `text-image-classification/src/models/late_fusion.py`, `text-image-classification/train_qmf.py`, plus `text-image-classification/src/utils/utils.py`

## Retained components

- one classifier per representation;
- confidence `0.1 * logsumexp(logits)`;
- detached confidence weights in dynamic late fusion;
- individual-view losses plus a fused loss;
- cumulative per-sample loss history;
- adjacent-pair margin-ranking loss;
- zero target for equal-history pairs. PyTorch therefore assigns zero loss to those pairs. `tests/test_analysis.py` verifies this behavior.

## SoccerNet adaptations

- a temporal CNN replaces the upstream text/image encoders;
- each 2-fps frame is one training sample;
- the target contains 17 event classes plus background;
- class-weighted cross-entropy handles sparse event frames;
- temporal clips replace shuffled independent examples;
- model selection uses validation loss;
- evaluation uses SoccerNet temporal NMS plus official spotting metrics;
- pooled isotonic calibration is added only when QMF confidence serves as a selector.

The port uses the text-image implementation's history-update order plus unit weighting of the two ranking losses. These choices match the pinned reference file. They differ from the separate RGB-D script, which updates history after computing the ranking loss plus multiplies ranking terms by `0.1`.
