"""RQ3 diagnosis: why does the class-conditional isotonic stage (Eq. 3) lower rare-class recall, and can it be fixed?
Hypothesis: the combiner and the isotonic maps are both fitted on the validation candidates; for a rare class the
class-specific coefficients nearly separate its few validation candidates, the in-sample scores are over-confident,
and the per-class isotonic map fitted on them is too steep. Remedy tested: cross-fitting (calibrate on out-of-fold
combiner scores), plus a shrunk class-offset alternative. Usage: python src/rq3_study.py --seed k  -> results/rq3_s<k>.json
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from analysis import *
from run_pipeline import available_games, tune_omega, f1_optimal_threshold, fit_reliability, select_combiner, class_priors, FULL, design, grouped_splits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit

def oof_scores(rows, feats, ci, C, n_splits=5):
    X = design(rows, feats, ci); y = np.array([r["correct"] for r in rows]); mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd
    oof = np.zeros(len(rows))
    for tr, te in grouped_splits(rows, n_splits):
        oof[te] = LogisticRegression(max_iter=3000, C=C).fit(Xs[tr], y[tr]).predict_proba(Xs[te])[:, 1]
    g = LogisticRegression(max_iter=3000, C=C).fit(Xs, y); ins = g.predict_proba(Xs)[:, 1]
    def raw(rows2): return g.predict_proba((design(rows2, feats, ci) - mu) / sd)[:, 1]
    return oof, ins, raw

def main(a):
    s = a.seed; prior, counts = class_priors(); rare = [c for c in range(NUM_CLASSES) if prior[c] < RARE_PRIOR]
    tags = [f"visual_s{s}", f"audio_s{s}"]; GV, GT = GameSet(available_games("valid", tags)), GameSet(available_games("test", tags))
    Pv_v, Pa_v = [sigmoid(load_logits(f"visual_s{s}", g, h)) for g in GV.games for h in (1, 2)], [sigmoid(load_logits(f"audio_s{s}", g, h)) for g in GV.games for h in (1, 2)]
    Pv_t, Pa_t = [sigmoid(load_logits(f"visual_s{s}", g, h)) for g in GT.games for h in (1, 2)], [sigmoid(load_logits(f"audio_s{s}", g, h)) for g in GT.games for h in (1, 2)]
    omega = tune_omega(GV, Pv_v, Pa_v); Pf_v, Pf_t = late_fusion(Pv_v, Pa_v, omega), late_fusion(Pv_t, Pa_t, omega)
    tau0, _ = f1_optimal_threshold(GV, Pf_v); rows_v, rows_t = build_candidates(GV, Pf_v, Pv_v, Pa_v, thr=tau0), build_candidates(GT, Pf_t, Pv_t, Pa_t, thr=tau0)
    cfg = select_combiner(rows_v, FULL); CI, CC = cfg["class_inter"], cfg["C"]
    yv, yt = np.array([r["correct"] for r in rows_v]), np.array([r["correct"] for r in rows_t]); cv, ct = np.array([r["cls"] for r in rows_v]), np.array([r["cls"] for r in rows_t])
    rv_mask, rt_mask = np.isin(cv, rare), np.isin(ct, rare)
    oof, ins, raw = oof_scores(rows_v, FULL, CI, CC); st = raw(rows_t)
    res = {"seed": s, "n_rare_val": int(rv_mask.sum()), "n_rare_test": int(rt_mask.sum()), "n_rare_test_correct": int((yt[rt_mask] == 1).sum()), "combiner": {"class_inter": CI, "C": CC}}
    # --- diagnosis: in-sample vs out-of-fold combiner scores on the rare validation candidates
    def auc(y, sc): return float(roc_auc_score(y, sc)) if len(np.unique(y)) > 1 else float("nan")
    res["rare_val_auc_insample"] = auc(yv[rv_mask], ins[rv_mask]); res["rare_val_auc_oof"] = auc(yv[rv_mask], oof[rv_mask])
    res["rare_val_mean_score_insample_correct"] = float(ins[rv_mask & (yv == 1)].mean()); res["rare_val_mean_score_oof_correct"] = float(oof[rv_mask & (yv == 1)].mean())
    res["rare_val_mean_score_insample_wrong"] = float(ins[rv_mask & (yv == 0)].mean()) if (rv_mask & (yv == 0)).any() else float("nan"); res["rare_val_mean_score_oof_wrong"] = float(oof[rv_mask & (yv == 0)].mean()) if (rv_mask & (yv == 0)).any() else float("nan")
    res["all_val_auc_insample"] = auc(yv, ins); res["all_val_auc_oof"] = auc(yv, oof)
    # --- variants
    def rare_recall(sc, cov=0.8):
        o = np.argsort(-sc); keep = np.zeros(len(sc), bool); keep[o[:int(cov * len(sc))]] = True; m = rt_mask & (yt == 1); return float(keep[m].mean())
    def evaluate(name, r):
        res[name] = {"rare_recall80": rare_recall(r), "rare_recall90": rare_recall(r, 0.9), "AURC": risk_coverage(r, yt)[2], "rare_ece": ece(r[rt_mask], yt[rt_mask]), "rare_mean_r": float(r[rt_mask].mean()), "rare_precision": float(yt[rt_mask].mean()),
                     "rare_rank_percentile_correct": float(np.mean([(r < x).mean() for x in r[rt_mask & (yt == 1)]]))}
    for fit_on, tag in [(ins, "insample"), (oof, "crossfit")]:
        for cc in (False, True):
            cal = Reliability(30, 20.0).fit(fit_on, yv, cv); evaluate(f"iso_{'classcond' if cc else 'pooled'}_{tag}", cal.predict(st, ct, class_conditional=cc))
        # class offset on the logit of the pooled map, MAP with Gaussian prior (precision kappa), estimated on the same scores
        pooled = Reliability(30, 20.0).fit(fit_on, yv, cv); rp_v = np.clip(pooled.predict(fit_on, cv, False), 1e-4, 1 - 1e-4); rp_t = np.clip(pooled.predict(st, ct, False), 1e-4, 1 - 1e-4)
        for kappa in (5.0, 20.0):
            r = rp_t.copy()
            for c in np.unique(cv):
                m = cv == c; z = logit(rp_v[m]); d = 0.0
                for _ in range(30):
                    p = expit(z + d); grad = (yv[m] - p).sum() - kappa * d; hess = -(p * (1 - p)).sum() - kappa; d -= grad / hess
                r[ct == c] = expit(logit(rp_t[ct == c]) + d)
            evaluate(f"offset_kappa{kappa:g}_{tag}", r)
    # rare-class-only pooling: one shared curve for all rare classes (n pooled over rare classes)
    for fit_on, tag in [(ins, "insample"), (oof, "crossfit")]:
        cal = Reliability(30, 20.0).fit(fit_on, yv, np.where(rv_mask, -1, cv)); evaluate(f"iso_classcond_rarepooled_{tag}", cal.predict(st, np.where(rt_mask, -1, ct), True))
    json.dump(res, open(os.path.join(RESULTS, f"rq3_s{s}.json"), "w"), indent=2)
    print("seed", s, "rare val", res["n_rare_val"], "AUC in-sample %.3f oof %.3f" % (res["rare_val_auc_insample"], res["rare_val_auc_oof"]))
    for k, v in res.items():
        if isinstance(v, dict) and "rare_recall80" in v: print(f"  {k:36s} recall80 {v['rare_recall80']:.3f} recall90 {v['rare_recall90']:.3f} AURC {v['AURC']:.4f} rareECE {v['rare_ece']:.3f} mean_r {v['rare_mean_r']:.2f} prec {v['rare_precision']:.2f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, required=True); main(p.parse_args())
