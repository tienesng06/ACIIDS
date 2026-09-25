"""Part 2 of the negative-result study: score-multiset-preserving reranking (keeps the per-class score histogram, changes only
the order, so the threshold-sampled official AP measures ordering and not score scale) and a per-feature ablation of
the temporal agreement features. Usage: python src/rerank_study2.py --seed 0
"""
import os, sys, json, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from analysis import *
from run_pipeline import available_games, tune_omega, f1_optimal_threshold, class_priors
from rerank_study import extra_features, loose_credit
from sklearn.linear_model import LogisticRegression

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def rerank_score_multiset(gs, Pf, rows, score):
    """Per class: candidates keep the multiset of their p_f values, reassigned in the order of `score`."""
    out = [nms_detections(P) for P in Pf]; idx = {}; k = 0
    for g in gs.games:
        for h in (1, 2): idx[(g, h)] = k; k += 1
    cls = np.array([r["cls"] for r in rows]); pf = np.array([r["p_f"] for r in rows]); score = np.asarray(score)
    for c in np.unique(cls):
        m = np.where(cls == c)[0]; vals = np.sort(pf[m])[::-1]; order = m[np.argsort(-score[m], kind="stable")]
        for i, v in zip(order, vals): out[idx[(rows[i]["game"], rows[i]["half"])]][rows[i]["frame"], c] = v
    return out

def main(a):
    s = a.seed; tags = [f"visual_s{s}", f"audio_s{s}"]
    val_games, test_games = available_games("valid", tags), available_games("test", tags)
    if a.fast: val_games, test_games = val_games[:20], test_games[:20]
    GV, GT = GameSet(val_games), GameSet(test_games)
    Pv_v, Pa_v = [sigmoid(load_logits(f"visual_s{s}", g, h)) for g in GV.games for h in (1, 2)], [sigmoid(load_logits(f"audio_s{s}", g, h)) for g in GV.games for h in (1, 2)]
    Pv_t, Pa_t = [sigmoid(load_logits(f"visual_s{s}", g, h)) for g in GT.games for h in (1, 2)], [sigmoid(load_logits(f"audio_s{s}", g, h)) for g in GT.games for h in (1, 2)]
    omega = tune_omega(GV, Pv_v, Pa_v); Pf_v, Pf_t = late_fusion(Pv_v, Pa_v, omega), late_fusion(Pv_t, Pa_t, omega)
    tau0, _ = f1_optimal_threshold(GV, Pf_v); rows_v, rows_t = build_candidates(GV, Pf_v, Pv_v, Pa_v, thr=tau0), build_candidates(GT, Pf_t, Pv_t, Pa_t, thr=tau0)
    idx_v, idx_t = {}, {}
    for idx, gs in ((idx_v, GV), (idx_t, GT)):
        k = 0
        for g in gs.games:
            for h in (1, 2): idx[(g, h)] = k; k += 1
    FULL = ["p_v", "p_a", "A_gap", "A_js", "A_prod", "p_f"]; EXTRA = ["dt_peak", "p_a_5s", "p_v_5s", "p_a_mean5s", "p_v_mean5s"]
    Xv, Xt = np.array([[r[k] for k in FULL] for r in rows_v]), np.array([[r[k] for k in FULL] for r in rows_t])
    Ev, Et = extra_features(rows_v, Pv_v, Pa_v, idx_v), extra_features(rows_t, Pv_t, Pa_t, idx_t)
    yv, yt = np.array([r["correct"] for r in rows_v]), np.array([r["correct"] for r in rows_t]); cv, ct = np.array([r["cls"] for r in rows_v]), np.array([r["cls"] for r in rows_t])
    def mAPs(D): lo, ti = GT.mAP(D, "loose"), GT.mAP(D, "tight"); return {"loose": round(lo["mAP"], 4), "tight": round(ti["mAP"], 4), "unshown": round(lo["unshown"], 4)}
    def fit(Xa, Xb): g = LogisticRegression(max_iter=2000).fit(Xa, yv); return g.predict_proba(Xa)[:, 1], g.predict_proba(Xb)[:, 1]
    def aurc_cc(sv, st): return round(risk_coverage(Reliability(30, 20.0).fit(sv, yv, cv).predict(st, ct, True), yt)[2], 4)
    res = {"seed": s}
    res["base"] = mAPs([nms_detections(P) for P in Pf_t]); log("base", res["base"])
    gv6, gt6 = fit(Xv, Xt); res["rp_rerank_g6"] = mAPs(rerank_score_multiset(GT, Pf_t, rows_t, gt6)); log("score-multiset rerank by g (6 feats)", res["rp_rerank_g6"])
    Xv2, Xt2 = np.hstack([Xv, Ev]), np.hstack([Xt, Et]); gv11, gt11 = fit(Xv2, Xt2)
    res["rp_rerank_g11"] = mAPs(rerank_score_multiset(GT, Pf_t, rows_t, gt11)); log("score-multiset rerank by g (11 feats)", res["rp_rerank_g11"])
    r11 = Reliability(30, 20.0).fit(gv11, yv, cv).predict(gt11, ct, True)
    res["rp_rerank_r11_tiebreak"] = mAPs(rerank_score_multiset(GT, Pf_t, rows_t, r11 + 1e-6 * gt11)); log("score-multiset rerank by r (11 feats, tie-broken)", res["rp_rerank_r11_tiebreak"])
    res["rp_rerank_pf_sanity"] = mAPs(rerank_score_multiset(GT, Pf_t, rows_t, Xt[:, -1])); log("sanity: score-multiset rerank with p_f itself", res["rp_rerank_pf_sanity"])
    # loose-credit-trained combiner with 11 feats
    lc_v = loose_credit(GV, rows_v)
    gl = LogisticRegression(max_iter=2000).fit(np.repeat(Xv2, 2, axis=0), np.concatenate([np.ones(len(Xv2)), np.zeros(len(Xv2))]), sample_weight=np.concatenate([lc_v, 1 - lc_v]))
    res["rp_rerank_g11_loose"] = mAPs(rerank_score_multiset(GT, Pf_t, rows_t, gl.predict_proba(Xt2)[:, 1])); log("score-multiset rerank by g (11 feats, loose-credit label)", res["rp_rerank_g11_loose"])
    # per-feature ablation of the temporal features (AURC, class-cond calibrated)
    abl = {"6 feats": aurc_cc(gv6, gt6), "11 feats": aurc_cc(gv11, gt11)}
    for j, nm in enumerate(EXTRA):
        a_, b_ = fit(np.hstack([Xv, Ev[:, [j]]]), np.hstack([Xt, Et[:, [j]]])); abl["6 + " + nm] = aurc_cc(a_, b_)
        keep = [i for i in range(len(EXTRA)) if i != j]; a_, b_ = fit(np.hstack([Xv, Ev[:, keep]]), np.hstack([Xt, Et[:, keep]])); abl["11 - " + nm] = aurc_cc(a_, b_)
    pv, pt = Xv[:, -1], Xt[:, -1]; abl["p_f only"] = aurc_cc(pv, pt)
    res["AURC_ablation"] = abl; log("AURC ablation", abl)
    # what the peak-offset looks like for correct vs wrong candidates
    res["dt_peak_mean_correct_vs_wrong"] = [float(Et[yt == 1, 0].mean()), float(Et[yt == 0, 0].mean())]
    res["p_a_5s_mean_correct_vs_wrong"] = [float(Et[yt == 1, 1].mean()), float(Et[yt == 0, 1].mean())]; res["p_a_1s_mean_correct_vs_wrong"] = [float(Xt[yt == 1, 1].mean()), float(Xt[yt == 0, 1].mean())]
    log("dt_peak", res["dt_peak_mean_correct_vs_wrong"], "p_a 5s", res["p_a_5s_mean_correct_vs_wrong"], "p_a 1s", res["p_a_1s_mean_correct_vs_wrong"])
    json.dump(res, open(os.path.join(RESULTS, f"study2_s{s}.json"), "w"), indent=2); log("written")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=0); p.add_argument("--fast", action="store_true"); main(p.parse_args())
