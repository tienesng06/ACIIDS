"""Optimisation study (seed 0): can the abstention and reranking gaps be widened legitimately?
All choices are made on the VALIDATION split by 5-fold cross-validation (calibrator fitted inside each fold); the test
split is evaluated once per configuration for reporting. Usage: python src/optimize_study.py --seed 0
"""
import os, sys, json, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from analysis import *
from run_pipeline import available_games, tune_omega, f1_optimal_threshold, AGREE6, FULL, grouped_splits
from sklearn.linear_model import LogisticRegression

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def window_feats(rows, Pv, Pa, Pf, idx, halves=(5, 10, 20, 40)):
    """Multi-scale context: max and mean of each branch over +-h frames; local prominence of p_f; same-class candidate density."""
    names, X = [], []
    for h in halves:
        for b in ("v", "a"): names += [f"max{b}_{h}", f"mean{b}_{h}"]
    names += ["prom_f_60", "dens_30", "dens_120", "rank_in_half"]
    by_half = {}
    for i, r in enumerate(rows): by_half.setdefault((r["game"], r["half"], r["cls"]), []).append(r["frame"])
    for r in rows:
        k = idx[(r["game"], r["half"])]; f, c = r["frame"], r["cls"]; n = Pv[k].shape[0]; row = []
        for h in halves:
            lo, hi = max(0, f - h), min(n, f + h + 1)
            for P in (Pv, Pa): w = P[k][lo:hi, c]; row += [float(w.max()), float(w.mean())]
        lo, hi = max(0, f - 60), min(n, f + 61); row.append(float(r["p_f"] - Pf[k][lo:hi, c].mean()))
        fr = by_half[(r["game"], r["half"], c)]
        row.append(float(sum(1 for x in fr if 0 < abs(x - f) <= 30))); row.append(float(sum(1 for x in fr if 0 < abs(x - f) <= 120)))
        row.append(float(sum(1 for x in fr if True)))  # number of same-class candidates in the half
        X.append(row)
    return names, np.array(X)

def cv_aurc(X, y, c, rows, n_splits=5, C=1.0, class_inter=False):
    """5-fold match-grouped CV: logistic plus class-conditional isotonic inside each fold."""
    oof = np.zeros(len(y))
    for tr, te in grouped_splits(rows, n_splits):
        g = LogisticRegression(max_iter=3000, C=C).fit(X[tr], y[tr]); s_tr, s_te = g.predict_proba(X[tr])[:, 1], g.predict_proba(X[te])[:, 1]
        oof[te] = Reliability(30, 20.0).fit(s_tr, y[tr], c[tr]).predict(s_te, c[te], True)
    return risk_coverage(oof, y)[2]

def with_class_inter(X, c, n_cls=NUM_CLASSES):
    oh = np.eye(n_cls)[c]; return np.hstack([X, oh, np.einsum("ij,ik->ijk", X, oh).reshape(len(X), -1)])

def main(a):
    s = a.seed; tags = [f"visual_s{s}", f"audio_s{s}"]
    val_games, test_games = available_games("valid", tags), available_games("test", tags)
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
    yv, yt = np.array([r["correct"] for r in rows_v]), np.array([r["correct"] for r in rows_t]); cv, ct = np.array([r["cls"] for r in rows_v]), np.array([r["cls"] for r in rows_t])
    Xv11, Xt11 = np.array([[r[k] for k in FULL] for r in rows_v]), np.array([[r[k] for k in FULL] for r in rows_t])
    wn, Wv = window_feats(rows_v, Pv_v, Pa_v, Pf_v, idx_v); _, Wt = window_feats(rows_t, Pv_t, Pa_t, Pf_t, idx_t)
    res = {"seed": s, "n_val": len(rows_v), "n_test": len(rows_t)}
    def test_aurc(Xa, Xb, C=1.0):
        g = LogisticRegression(max_iter=3000, C=C).fit(Xa, yv); r = Reliability(30, 20.0).fit(g.predict_proba(Xa)[:, 1], yv, cv).predict(g.predict_proba(Xb)[:, 1], ct, True)
        return risk_coverage(r, yt)[2], g
    configs = {}
    configs["A: 11 feats (current)"] = (Xv11, Xt11)
    ms = [i for i, n in enumerate(wn) if n.startswith(("max", "mean"))]
    configs["B: 11 + multiscale windows"] = (np.hstack([Xv11, Wv[:, ms]]), np.hstack([Xt11, Wt[:, ms]]))
    dn = [i for i, n in enumerate(wn) if n.startswith(("prom", "dens", "rank"))]
    configs["C: 11 + prominence/density"] = (np.hstack([Xv11, Wv[:, dn]]), np.hstack([Xt11, Wt[:, dn]]))
    configs["D: 11 + multiscale + density"] = (np.hstack([Xv11, Wv]), np.hstack([Xt11, Wt]))
    configs["E: D + class interactions"] = (with_class_inter(np.hstack([Xv11, Wv]), cv), with_class_inter(np.hstack([Xt11, Wt]), ct))
    configs["F: 11 + class interactions"] = (with_class_inter(Xv11, cv), with_class_inter(Xt11, ct))
    out = {}
    for nm, (Xa, Xb) in configs.items():
        # standardise for the regularised fit
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-9; Xa_, Xb_ = (Xa - mu) / sd, (Xb - mu) / sd
        best = None
        for C in (0.03, 0.1, 0.3, 1.0, 3.0):
            v = cv_aurc(Xa_, yv, cv, rows_v, C=C)
            if best is None or v < best[0]: best = (v, C)
        ta, g = test_aurc(Xa_, Xb_, C=best[1])
        out[nm] = {"cv_val_AURC": round(best[0], 4), "C": best[1], "test_AURC": round(ta, 4), "n_feats": int(Xa.shape[1])}; log(nm, out[nm])
    res["abstention"] = out
    # ---------- reranking: wider candidate set for reranking, class-interaction combiner ----------
    def mAPs(D): lo, ti = GT.mAP(D, "loose"), GT.mAP(D, "tight"); return {"loose": round(lo["mAP"], 4), "tight": round(ti["mAP"], 4), "unshown": round(lo["unshown"], 4)}
    def vmAP(D): return round(GV.mAP(D, "loose")["mAP"], 4)
    base_t = [nms_detections(P) for P in Pf_t]; base_v = [nms_detections(P) for P in Pf_v]
    rr = {"base": {"val_loose": vmAP(base_v), **mAPs(base_t)}}; log("rerank base", rr["base"])
    for thr in (tau0, 0.2, 0.1):
        rv2 = rows_v if thr == tau0 else build_candidates(GV, Pf_v, Pv_v, Pa_v, thr=thr); rt2 = rows_t if thr == tau0 else build_candidates(GT, Pf_t, Pv_t, Pa_t, thr=thr)
        yv2 = np.array([r["correct"] for r in rv2]); cv2 = np.array([r["cls"] for r in rv2]); ct2 = np.array([r["cls"] for r in rt2])
        Xa = np.array([[r[k] for k in FULL] for r in rv2]); Xb = np.array([[r[k] for k in FULL] for r in rt2])
        for nm, (A_, B_) in {"11 feats": (Xa, Xb), "11 + class inter": (with_class_inter(Xa, cv2), with_class_inter(Xb, ct2))}.items():
            mu, sd = A_.mean(0), A_.std(0) + 1e-9; A_, B_ = (A_ - mu) / sd, (B_ - mu) / sd
            g = LogisticRegression(max_iter=3000, C=0.3).fit(A_, yv2)
            # validation mAP uses an in-sample combiner (optimistic) -> use 2-fold OOF on validation for the choice
            oof = np.zeros(len(A_))
            for tr, te in grouped_splits(rv2, 2): oof[te] = LogisticRegression(max_iter=3000, C=0.3).fit(A_[tr], yv2[tr]).predict_proba(A_[te])[:, 1]
            key = f"thr={thr:g}, {nm}"; rr[key] = {"val_loose_oof": vmAP(rerank_score_multiset(GV, Pf_v, rv2, oof)), **mAPs(rerank_score_multiset(GT, Pf_t, rt2, g.predict_proba(B_)[:, 1]))}; log("rerank", key, rr[key])
    res["rerank"] = rr
    json.dump(res, open(os.path.join(RESULTS, f"optimize_s{s}.json"), "w"), indent=2); log("written")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=0); main(p.parse_args())
