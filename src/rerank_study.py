"""Diagnostic study for the two negative results (seed 0, test split), run after the main pipeline.
Q1 (reranking hurts mAP): is it (a) score ties from the isotonic step function, (b) the +-5 s label vs loose-mAP
    credit mismatch, or (c) the combiner itself not ranking better than p_f within class?
Q2 (agreement adds little over class-conditional calibration): does a richer combiner (gradient boosting, extra
    temporal-agreement features) raise the increment?
Q3 (rare classes): does a one-parameter class offset with shrinkage (estimable from ~20 candidates) change rare recall?
Usage: python src/rerank_study.py --seed 0 [--fast]
"""
import os, sys, json, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from analysis import *
from run_pipeline import available_games, tune_omega, f1_optimal_threshold, fit_reliability, apply_rerank, selective_table, class_priors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def loose_credit(gs, rows):
    """Fraction of the 12 loose tolerances (5..60 s) at which the candidate lies within delta of a same-class event."""
    out = []
    for r in rows:
        ev = [f for f, c, _ in gs.events[r["game"]][r["half"]] if c == r["cls"]]
        d = min([abs(f - r["frame"]) for f in ev], default=10 ** 9) / FPS   # seconds
        out.append(float(np.mean([d <= delta for delta in range(5, 65, 5)])))
    return np.array(out)

def extra_features(rows, Pv, Pa, idx):
    """Temporal agreement features: peak offset between branches within +-5 s, audio max within +-5 s, visual max within +-5 s."""
    X = []
    for r in rows:
        k = idx[(r["game"], r["half"])]; f, c = r["frame"], r["cls"]
        lo, hi = max(0, f - CORR_DELTA), min(Pv[k].shape[0], f + CORR_DELTA + 1)
        wv, wa = Pv[k][lo:hi, c], Pa[k][lo:hi, c]
        off = abs(int(np.argmax(wv)) - int(np.argmax(wa))) / FPS
        X.append([off, float(wa.max()), float(wv.max()), float(wa.mean()), float(wv.mean())])
    return np.array(X)

def rerank_by(gs, Pf, rows, score):
    """Sparse detections where candidates get `score` and non-candidates keep p_f scaled below every candidate."""
    out = [np.zeros_like(P) for P in Pf]; idx = {}; k = 0
    for g in gs.games:
        for h in (1, 2): idx[(g, h)] = k; k += 1
    smin = float(np.min(score)) if len(score) else 0.0
    for P, D in zip(Pf, out):
        D[:] = nms_detections(P); m = D >= 0; D[m] = D[m] * 1e-3   # below every candidate score
    for row, s in zip(rows, score): out[idx[(row["game"], row["half"])]][row["frame"], row["cls"]] = 1e-3 + float(s)
    return out

def main(a):
    s = a.seed; prior, counts = class_priors(); rare = [c for c in range(NUM_CLASSES) if prior[c] < RARE_PRIOR]
    tags = [f"visual_s{s}", f"audio_s{s}"]
    val_games, test_games = available_games("valid", tags), available_games("test", tags)
    if a.fast: val_games, test_games = val_games[:20], test_games[:20]
    GV, GT = GameSet(val_games), GameSet(test_games); log(f"valid {len(val_games)} test {len(test_games)} games")
    Pv_v, Pa_v = [sigmoid(load_logits(f"visual_s{s}", g, h)) for g in GV.games for h in (1, 2)], [sigmoid(load_logits(f"audio_s{s}", g, h)) for g in GV.games for h in (1, 2)]
    Pv_t, Pa_t = [sigmoid(load_logits(f"visual_s{s}", g, h)) for g in GT.games for h in (1, 2)], [sigmoid(load_logits(f"audio_s{s}", g, h)) for g in GT.games for h in (1, 2)]
    omega = tune_omega(GV, Pv_v, Pa_v); Pf_v, Pf_t = late_fusion(Pv_v, Pa_v, omega), late_fusion(Pv_t, Pa_t, omega)
    tau0, _ = f1_optimal_threshold(GV, Pf_v); rows_v, rows_t = build_candidates(GV, Pf_v, Pv_v, Pa_v, thr=tau0), build_candidates(GT, Pf_t, Pv_t, Pa_t, thr=tau0)
    log(f"omega={omega} tau0={tau0} candidates valid {len(rows_v)} test {len(rows_t)}")
    idx_v, idx_t = {}, {}
    for idx, gs in ((idx_v, GV), (idx_t, GT)):
        k = 0
        for g in gs.games:
            for h in (1, 2): idx[(g, h)] = k; k += 1
    FULL = ["p_v", "p_a", "A_gap", "A_js", "A_prod", "p_f"]
    Xv, Xt = np.array([[r[k] for k in FULL] for r in rows_v]), np.array([[r[k] for k in FULL] for r in rows_t])
    yv, yt = np.array([r["correct"] for r in rows_v]), np.array([r["correct"] for r in rows_t])
    cv, ct = np.array([r["cls"] for r in rows_v]), np.array([r["cls"] for r in rows_t])
    pf_t = Xt[:, -1]
    res = {"seed": s, "omega": omega, "tau0": tau0, "n_test": len(rows_t)}
    def mAPs(D): lo, ti = GT.mAP(D, "loose"), GT.mAP(D, "tight"); return {"loose": lo["mAP"], "tight": ti["mAP"], "unshown": lo["unshown"]}
    # ---------- Q1: reranking ----------
    base = [nms_detections(P) for P in Pf_t]; res["Q1_base_late_fusion"] = mAPs(base); log("Q1 base", res["Q1_base_late_fusion"])
    g = LogisticRegression(max_iter=1000).fit(Xv, yv); gt_ = g.predict_proba(Xt)[:, 1]
    sel_full = fit_reliability(rows_v, FULL, class_cond=True); r_t = sel_full(rows_t)
    res["Q1_ties_in_r"] = {"unique_values": int(len(np.unique(r_t))), "n": int(len(r_t)), "largest_tie_block": int(np.max(np.unique(r_t, return_counts=True)[1]))}
    res["Q1_rerank_r_classcond_isotonic"] = mAPs(rerank_by(GT, Pf_t, rows_t, r_t)); log("Q1 r (ties)", res["Q1_rerank_r_classcond_isotonic"])
    res["Q1_rerank_r_tiebreak_by_g"] = mAPs(rerank_by(GT, Pf_t, rows_t, r_t + 1e-6 * gt_)); log("Q1 r tie-broken", res["Q1_rerank_r_tiebreak_by_g"])
    res["Q1_rerank_g_logistic_5s"] = mAPs(rerank_by(GT, Pf_t, rows_t, gt_)); log("Q1 g (logistic, +-5 s label)", res["Q1_rerank_g_logistic_5s"])
    res["Q1_rerank_pf_r_lambda0.5_tiebreak"] = mAPs(rerank_by(GT, Pf_t, rows_t, pf_t * np.sqrt(r_t) + 1e-6 * gt_)); log("Q1 p_f r^0.5 tie-broken", res["Q1_rerank_pf_r_lambda0.5_tiebreak"])
    # (b) label mismatch: combiner trained on loose credit (fraction of tolerances matched) instead of the +-5 s label
    lc_v = loose_credit(GV, rows_v); lc_t = loose_credit(GT, rows_t)
    from sklearn.linear_model import LinearRegression
    gl = LogisticRegression(max_iter=1000).fit(np.repeat(Xv, 2, axis=0), np.concatenate([np.ones(len(Xv)), np.zeros(len(Xv))]), sample_weight=np.concatenate([lc_v, 1 - lc_v]))
    gl_t = gl.predict_proba(Xt)[:, 1]
    res["Q1_rerank_g_loose_credit"] = mAPs(rerank_by(GT, Pf_t, rows_t, gl_t)); log("Q1 g (loose-credit label)", res["Q1_rerank_g_loose_credit"])
    # (c) per-class ranking check: within-class AP of p_f vs g vs g_loose on the candidate set alone (label = loose credit >= 0.5 and = +-5 s)
    from sklearn.metrics import average_precision_score
    def within_class_ap(score, y):
        aps, ws = [], []
        for c in np.unique(ct):
            m = ct == c
            if y[m].sum() > 0 and (1 - y[m]).sum() > 0: aps.append(average_precision_score(y[m], score[m])); ws.append(m.sum())
        return float(np.average(aps, weights=ws))
    res["Q1_candidate_AP_5s"] = {"p_f": within_class_ap(pf_t, yt), "g": within_class_ap(gt_, yt), "r": within_class_ap(r_t, yt), "g_loose": within_class_ap(gl_t, yt)}
    y30 = (lc_t >= 0.5).astype(int)
    res["Q1_candidate_AP_30s"] = {"p_f": within_class_ap(pf_t, y30), "g": within_class_ap(gt_, y30), "r": within_class_ap(r_t, y30), "g_loose": within_class_ap(gl_t, y30)}
    log("Q1 within-class AP", res["Q1_candidate_AP_5s"], res["Q1_candidate_AP_30s"])
    # ---------- Q2: richer combiner ----------
    Ev, Et = extra_features(rows_v, Pv_v, Pa_v, idx_v), extra_features(rows_t, Pv_t, Pa_t, idx_t)
    Xv2, Xt2 = np.hstack([Xv, Ev]), np.hstack([Xt, Et])
    def aurc_of(score_v, score_t):
        cal = Reliability(n_min=30, shrink=20.0).fit(score_v, yv, cv); r = cal.predict(score_t, ct, class_conditional=True)
        return risk_coverage(r, yt)[2], r
    a_pf, _ = aurc_of(Xv[:, -1], Xt[:, -1]); a_g, _ = aurc_of(g.predict_proba(Xv)[:, 1], gt_)
    g2 = LogisticRegression(max_iter=2000).fit(Xv2, yv); a_g2, _ = aurc_of(g2.predict_proba(Xv2)[:, 1], g2.predict_proba(Xt2)[:, 1])
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(Xv, yv); a_gb, _ = aurc_of(gb.predict_proba(Xv)[:, 1], gb.predict_proba(Xt)[:, 1])
    gb2 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(Xv2, yv); a_gb2, r_gb2 = aurc_of(gb2.predict_proba(Xv2)[:, 1], gb2.predict_proba(Xt2)[:, 1])
    # in-sample calibration risk: GB fitted and calibrated on the same validation rows -> use 2-fold within validation for the calibrator
    from run_pipeline import grouped_splits
    oof = np.zeros(len(Xv2))
    for tr, te in grouped_splits(rows_v, 5):
        oof[te] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(Xv2[tr], yv[tr]).predict_proba(Xv2[te])[:, 1]
    cal = Reliability(n_min=30, shrink=20.0).fit(oof, yv, cv); r_gb2_oof = cal.predict(gb2.predict_proba(Xt2)[:, 1], ct, class_conditional=True)
    res["Q2_AURC"] = {"p_f class-cond": a_pf, "logistic 6 feats (paper)": a_g, "logistic +5 temporal feats": a_g2, "GB 6 feats": a_gb, "GB +5 temporal feats": a_gb2, "GB +5 feats, OOF-calibrated": risk_coverage(r_gb2_oof, yt)[2]}
    log("Q2", res["Q2_AURC"])
    res["Q2_rerank_gb2"] = mAPs(rerank_by(GT, Pf_t, rows_t, gb2.predict_proba(Xt2)[:, 1])); log("Q2 rerank by GB+feats", res["Q2_rerank_gb2"])
    # ---------- Q3: rare classes, one-parameter class offset with shrinkage ----------
    from scipy.special import logit, expit
    gv_ = g.predict_proba(Xv)[:, 1]
    pooled = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(gv_, yv)
    rp_v, rp_t = np.clip(pooled.predict(gv_), 1e-4, 1 - 1e-4), np.clip(pooled.predict(gt_), 1e-4, 1 - 1e-4)
    def offset_cal(kappa=20.0):
        r = rp_t.copy()
        for c in np.unique(cv):
            m = cv == c; n = m.sum()
            if n == 0: continue
            # MLE of a logit offset with a Gaussian prior (shrinkage): one Newton step from 0 with prior precision kappa/n
            z = logit(rp_v[m]); d = 0.0
            for _ in range(20):
                p = expit(z + d); grad = (yv[m] - p).sum() - (kappa / 1.0) * d; hess = -(p * (1 - p)).sum() - kappa
                d -= grad / hess
            mt = ct == c; r[mt] = expit(logit(rp_t[mt]) + d)
        return r
    r_off = offset_cal()
    def rare_recall(sc, cov=0.8):
        o = np.argsort(-sc); keep = np.zeros(len(sc), bool); keep[o[:int(cov * len(sc))]] = True
        m = np.isin(ct, rare) & (yt == 1); return float(keep[m].mean()) if m.any() else float("nan"), int(m.sum())
    res["Q3_rare_recall80"] = {"pooled r": rare_recall(rp_t)[0], "class-cond isotonic (paper)": rare_recall(r_t)[0], "class offset, kappa=20": rare_recall(r_off)[0], "n_rare_correct": rare_recall(r_t)[1]}
    res["Q3_AURC"] = {"pooled r": risk_coverage(rp_t, yt)[2], "class offset": risk_coverage(r_off, yt)[2], "class-cond isotonic": risk_coverage(r_t, yt)[2]}
    m = np.isin(ct, rare); res["Q3_rare_mean_r_vs_precision"] = {"pooled": float(rp_t[m].mean()), "offset": float(r_off[m].mean()), "isotonic": float(r_t[m].mean()), "precision": float(yt[m].mean())}
    log("Q3", res["Q3_rare_recall80"], res["Q3_AURC"], res["Q3_rare_mean_r_vs_precision"])
    ensure_dirs(); json.dump(res, open(os.path.join(RESULTS, f"study_s{s}.json"), "w"), indent=2); log("written", f"results/study_s{s}.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=0); p.add_argument("--fast", action="store_true"); main(p.parse_args())
