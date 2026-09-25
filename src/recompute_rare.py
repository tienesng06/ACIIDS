"""Recompute results/<dir>/rare_s<k>.csv for one seed with the v5 definitions (AGREE pooled = recommended estimator,
AGREE class-conditional = Eq. (3) ablation); used to repair the v5 run whose rare rows duplicated the pooled selector."""
import os, sys, argparse, csv
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from analysis import *
from run_pipeline import available_games, tune_omega, f1_optimal_threshold, fit_reliability, select_combiner, class_priors, FULL, write_csv

def main(a):
    s = a.seed; prior, counts = class_priors(); rare = [c for c in range(NUM_CLASSES) if prior[c] < RARE_PRIOR]
    tags = [f"visual_s{s}", f"audio_s{s}"]; GV, GT = GameSet(available_games("valid", tags)), GameSet(available_games("test", tags))
    Pv_v, Pa_v = [sigmoid(load_logits(f"visual_s{s}", g, h)) for g in GV.games for h in (1, 2)], [sigmoid(load_logits(f"audio_s{s}", g, h)) for g in GV.games for h in (1, 2)]
    Pv_t, Pa_t = [sigmoid(load_logits(f"visual_s{s}", g, h)) for g in GT.games for h in (1, 2)], [sigmoid(load_logits(f"audio_s{s}", g, h)) for g in GT.games for h in (1, 2)]
    omega = tune_omega(GV, Pv_v, Pa_v); Pf_v, Pf_t = late_fusion(Pv_v, Pa_v, omega), late_fusion(Pv_t, Pa_t, omega)
    tau0, _ = f1_optimal_threshold(GV, Pf_v); rows_v, rows_t = build_candidates(GV, Pf_v, Pv_v, Pa_v, thr=tau0), build_candidates(GT, Pf_t, Pv_t, Pa_t, thr=tau0)
    cfg = select_combiner(rows_v, FULL); CI, CC = cfg["class_inter"], cfg["C"]
    sels = {"Confidence-only": fit_reliability(rows_v, ["p_f"], False)(rows_t), "Confidence-only class-conditional": fit_reliability(rows_v, ["p_f"], True)(rows_t),
            "AGREE pooled": fit_reliability(rows_v, FULL, False, class_inter=CI, C=CC)(rows_t), "AGREE class-conditional": fit_reliability(rows_v, FULL, True, class_inter=CI, C=CC)(rows_t)}
    y_t = np.array([r["correct"] for r in rows_t]); cls_arr = np.array([r["cls"] for r in rows_t]); rare_mask = np.isin(cls_arr, rare)
    def rare_recall(sc, cov=0.8):
        o = np.argsort(-sc); keep = np.zeros(len(sc), bool); keep[o[:int(cov * len(sc))]] = True
        m = rare_mask & (y_t == 1); return float(keep[m].mean()) if m.any() else float("nan"), int(m.sum())
    rows = []
    for name, sc in sels.items():
        rr, n_r = rare_recall(sc); rr90, _ = rare_recall(sc, 0.9)
        rows.append({"seed": s, "selector": name, "rare_recall80": rr, "rare_recall90": rr90, "n_rare_correct": n_r, "rare_ece": ece(sc[rare_mask], y_t[rare_mask]) if rare_mask.any() else float("nan"),
                     "rare_mean_r": float(sc[rare_mask].mean()) if rare_mask.any() else float("nan"), "rare_precision": float(y_t[rare_mask].mean()) if rare_mask.any() else float("nan")})
    write_csv(os.path.join(RESULTS, a.dir, f"rare_s{s}.csv"), rows); print("seed", s, [(r["selector"], round(r["rare_recall80"], 3)) for r in rows], "penalty val cands", sum(1 for r in rows_v if r["cls"] in rare))

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, required=True); p.add_argument("--dir", default="v5"); main(p.parse_args())
