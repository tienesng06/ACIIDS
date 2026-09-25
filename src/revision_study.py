"""Round-1 review follow-ups on seed 0 (test split): (1) paired-bootstrap CIs for AGREE with POOLED calibration on top of
the class-specific combiner; (2) per-half score-multiset-preserving reranking vs the test-set-wide multiset;
(3) reranker feature ablation (visual context only; visual-only, no audio); (4) how many candidates the first reranker
p_f r^0.5 pushed below tau0. Writes results/revision_s0.json. Usage: python src/revision_study.py --seed 0
"""
import os, sys, json, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from analysis import *
from run_pipeline import available_games, tune_omega, f1_optimal_threshold, fit_reliability, FULL, CTX_V, apply_rerank

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def rerank_per_half(gs, Pf, rows, score):
    """Score-multiset-preserving reranking within each (game, half, class)."""
    out = [nms_detections(P) for P in Pf]; idx = {}; k = 0
    for g in gs.games:
        for h in (1, 2): idx[(g, h)] = k; k += 1
    key = np.array([idx[(r["game"], r["half"])] * 100 + r["cls"] for r in rows]); pf = np.array([r["p_f"] for r in rows]); score = np.asarray(score, float)
    for kk in np.unique(key):
        m = np.where(key == kk)[0]; vals = np.sort(pf[m])[::-1]; order = m[np.argsort(-score[m], kind="stable")]
        for i, v in zip(order, vals): out[idx[(rows[i]["game"], rows[i]["half"])]][rows[i]["frame"], rows[i]["cls"]] = v
    return out

def main(a):
    s = a.seed; S = json.load(open(os.path.join(RESULTS, "summary.json"))); cb = S["combiner_by_seed"][str(s)]; rb = S["rerank_by_seed"][str(s)]
    tags = [f"visual_s{s}", f"audio_s{s}"]; GV, GT = GameSet(available_games("valid", tags)), GameSet(available_games("test", tags))
    Pv_v, Pa_v = [sigmoid(load_logits(f"visual_s{s}", g, h)) for g in GV.games for h in (1, 2)], [sigmoid(load_logits(f"audio_s{s}", g, h)) for g in GV.games for h in (1, 2)]
    Pv_t, Pa_t = [sigmoid(load_logits(f"visual_s{s}", g, h)) for g in GT.games for h in (1, 2)], [sigmoid(load_logits(f"audio_s{s}", g, h)) for g in GT.games for h in (1, 2)]
    omega = tune_omega(GV, Pv_v, Pa_v); Pf_v, Pf_t = late_fusion(Pv_v, Pa_v, omega), late_fusion(Pv_t, Pa_t, omega)
    tau0, _ = f1_optimal_threshold(GV, Pf_v); rows_v, rows_t = build_candidates(GV, Pf_v, Pv_v, Pa_v, thr=tau0), build_candidates(GT, Pf_t, Pv_t, Pa_t, thr=tau0)
    yt = np.array([r["correct"] for r in rows_t]); res = {"seed": s, "tau0": tau0, "omega": omega}
    # (1) pooled-calibration AGREE vs confidence-only: CIs
    sel_full_cc = fit_reliability(rows_v, FULL, True, class_inter=cb["class_inter"], C=cb["C"]); sel_full_pool = fit_reliability(rows_v, FULL, False, class_inter=cb["class_inter"], C=cb["C"])
    sel_conf = fit_reliability(rows_v, ["p_f"], False); sel_conf_cc = fit_reliability(rows_v, ["p_f"], True)
    r_pool, r_cc, r_conf, r_confcc = sel_full_pool(rows_t), sel_full_cc(rows_t), sel_conf(rows_t), sel_conf_cc(rows_t)
    for r, v in zip(rows_t, r_pool): r["r_pool"] = float(v)
    for r, v in zip(rows_t, r_conf): r["r_conf"] = float(v)
    for r, v in zip(rows_t, r_confcc): r["r_conf_cc"] = float(v)
    def au(sub, key): y = np.array([r["correct"] for r in sub]); return risk_coverage(np.array([r[key] for r in sub]), y)[2]
    ci = {}
    for name, fn in [("aurc_diff_pool_vs_conf", lambda sub: au(sub, "r_conf") - au(sub, "r_pool")), ("aurc_rel_pool_vs_conf", lambda sub: 100 * (au(sub, "r_conf") - au(sub, "r_pool")) / au(sub, "r_conf")),
                     ("aurc_diff_pool_vs_confcc", lambda sub: au(sub, "r_conf_cc") - au(sub, "r_pool")), ("aurc_rel_pool_vs_confcc", lambda sub: 100 * (au(sub, "r_conf_cc") - au(sub, "r_pool")) / au(sub, "r_conf_cc"))]:
        ci[name] = list(bootstrap_by_game(rows_t, fn, n_boot=a.n_boot)); log(name, ci[name])
    res["ci_pooled"] = ci; res["aurc_seed"] = {"conf": risk_coverage(r_conf, yt)[2], "conf_cc": risk_coverage(r_confcc, yt)[2], "agree_pool": risk_coverage(r_pool, yt)[2], "agree_cc": risk_coverage(r_cc, yt)[2]}
    # (4) first reranker: fraction of candidates pushed below tau0
    s_old = np.array([r["p_f"] for r in rows_t]) * np.sqrt(np.maximum(r_cc, 1e-3)); res["first_reranker_frac_below_tau0"] = float((s_old < tau0).mean()); log("frac below tau0", res["first_reranker_frac_below_tau0"])
    def mAPs(D): lo, ti = GT.mAP(D, "loose"), GT.mAP(D, "tight"); return {"loose": lo["mAP"], "tight": ti["mAP"], "unshown": lo["unshown"]}
    res["late_fusion"] = mAPs([nms_detections(P) for P in Pf_t])
    res["first_reranker_apply_rerank"] = mAPs(apply_rerank(GT, Pf_t, rows_t, r_cc, 0.5))
    # candidates re-scored by p_f r^0.5 while remaining above every non-candidate
    res["first_reranker_score_multiset_control"] = mAPs(rerank_score_multiset(GT, Pf_t, rows_t, s_old)); log("first reranker", res["first_reranker_apply_rerank"], "its score-multiset control", res["first_reranker_score_multiset_control"])
    # (2)+(3) reranking all detections >= tau_r: test-set-wide vs per-half; feature ablations
    rows_v_r, rows_t_r = build_candidates(GV, Pf_v, Pv_v, Pa_v, thr=rb["thr"]), build_candidates(GT, Pf_t, Pv_t, Pa_t, thr=rb["thr"])
    for nm, feats in [("all 11 (ours)", FULL), ("p_f + visual context", ["p_f"] + CTX_V), ("visual only (p_v + visual context)", ["p_v"] + CTX_V), ("p_f only (no reordering)", ["p_f"])]:
        sc = fit_reliability(rows_v_r, feats, True, class_inter=rb["class_inter"] and len(feats) > 1, C=rb["C"]).raw(rows_t_r)
        res[f"rerank_testwide[{nm}]"] = mAPs(rerank_score_multiset(GT, Pf_t, rows_t_r, sc)); res[f"rerank_perhalf[{nm}]"] = mAPs(rerank_per_half(GT, Pf_t, rows_t_r, sc))
        log(nm, "test-wide", res[f"rerank_testwide[{nm}]"], "per-half", res[f"rerank_perhalf[{nm}]"])
    json.dump(res, open(os.path.join(RESULTS, f"revision_s{s}.json"), "w"), indent=2); log("written")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=0); p.add_argument("--n-boot", type=int, default=1000); main(p.parse_args())
