"""Fix study for the unshown gap (NetVLAD++ 37.2 vs our late fusion 33.1 / AGREE rerank 35.0 loose unshown mAP).
Diagnosis (src/unshown_study.py): the gap sits in restart events that are mostly off-camera (indirect free-kick,
kick-off, clearance); NetVLAD++ trains on 15 s window-level labels, so it scores diffuse post-event context, while our
branches are trained on +-0.5 s targets. The post-hoc remedy gives the score-multiset-preserving
reranker wider, asymmetric temporal context of each branch (mean score over the 15 s / 30 s BEFORE and AFTER the
detection, and the max over the window), as extra combiner features, selected on validation like the rest.
Writes results/unshown_fix_s{seed}.json.
"""
import os, sys, json, argparse, time
import numpy as np
from scipy.ndimage import maximum_filter1d
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from analysis import *
from run_pipeline import available_games, tune_omega, FULL, select_reranker, fit_reliability, select_combiner, selective_table

def pooled_arrays(P_list, W, prefix):
    """Per-half (n, C) arrays: mean over [t-W, t-1] (pre), mean over [t+1, t+W] (post), max over [t-W, t+W]."""
    out = {f"{prefix}_pre{W//FPS}": [], f"{prefix}_post{W//FPS}": [], f"{prefix}_max{W//FPS}": []}
    for P in P_list:
        n = P.shape[0]; P64 = P.astype(np.float64); cs = np.vstack([np.zeros((1, P.shape[1])), np.cumsum(P64, axis=0)])
        t = np.arange(n)
        lo, hi = np.maximum(0, t - W), t                       # [t-W, t-1]
        pre = (cs[hi] - cs[lo]) / np.maximum(1, hi - lo)[:, None]
        lo2, hi2 = np.minimum(n, t + 1), np.minimum(n, t + W + 1)   # [t+1, t+W]
        post = (cs[hi2] - cs[lo2]) / np.maximum(1, hi2 - lo2)[:, None]
        mx = maximum_filter1d(P, size=2 * W + 1, axis=0, mode="nearest").astype(np.float32)
        out[f"{prefix}_pre{W//FPS}"].append(pre.astype(np.float32)); out[f"{prefix}_post{W//FPS}"].append(post.astype(np.float32)); out[f"{prefix}_max{W//FPS}"].append(mx)
    return out

def main(a):
    s = a.seed; t0 = time.time()
    tags = [f"visual_s{s}", f"audio_s{s}"]
    val_games, test_games = available_games("valid", tags), available_games("test", tags)
    GV, GT = GameSet(val_games), GameSet(test_games)
    Pv_v, Pa_v = [sigmoid(load_logits(tags[0], g, h)) for g in GV.games for h in (1, 2)], [sigmoid(load_logits(tags[1], g, h)) for g in GV.games for h in (1, 2)]
    Pv_t, Pa_t = [sigmoid(load_logits(tags[0], g, h)) for g in GT.games for h in (1, 2)], [sigmoid(load_logits(tags[1], g, h)) for g in GT.games for h in (1, 2)]
    omega = tune_omega(GV, Pv_v, Pa_v); Pf_v, Pf_t = late_fusion(Pv_v, Pa_v, omega), late_fusion(Pv_t, Pa_t, omega)
    print(f"valid {len(val_games)} test {len(test_games)} omega {omega}", flush=True)
    extra_v, extra_t = {}, {}
    for W in a.windows:
        extra_v.update(pooled_arrays(Pv_v, W * FPS, "p_v")); extra_v.update(pooled_arrays(Pa_v, W * FPS, "p_a"))
        extra_t.update(pooled_arrays(Pv_t, W * FPS, "p_v")); extra_t.update(pooled_arrays(Pa_t, W * FPS, "p_a"))
    # restart-structure context (post-hoc, from the fused score arrays and the clock): seconds since the half started, and the
    # max fused score of every class over the preceding minute (which event preceded this detection: goal -> kick-off,
    # foul -> free-kick, ball out -> throw-in / corner)
    def structural(Pf_list):
        ex = {"t_half_exp20": [], "t_half_cap600": []}
        for c in range(NUM_CLASSES): ex[f"prev60_{c}"] = []; ex[f"prev150_{c}"] = []
        for P in Pf_list:
            n = P.shape[0]; t = np.arange(n) / FPS
            ex["t_half_exp20"].append(np.exp(-t / 20.0).astype(np.float32)); ex["t_half_cap600"].append((np.minimum(t, 600.0) / 600.0).astype(np.float32))
            for W, key in ((60 * FPS, "prev60"), (150 * FPS, "prev150")):
                # max over [t-W, t-10 frames] (exclude the detection's own neighbourhood)
                mx = maximum_filter1d(P, size=W, axis=0, mode="nearest", origin=(W // 2) - 1)   # window [t-W+1, t]
                sh = np.vstack([np.zeros((10, P.shape[1]), P.dtype), mx[:-10]])                 # shifted 10 frames (5 s) back
                for c in range(NUM_CLASSES): ex[f"{key}_{c}"].append(sh[:, c].astype(np.float32))
        return ex
    extra_v.update(structural(Pf_v)); extra_t.update(structural(Pf_t))
    print(f"pooled arrays ready ({time.time()-t0:.0f}s): {sorted(extra_v)}", flush=True)
    ctx = {W: [f"p_{b}_{k}{W}" for b in ("v", "a") for k in ("pre", "post", "max")] for W in a.windows}
    feature_sets = [("FULL (11, current)", FULL)] + [(f"FULL + ctx{W}", FULL + ctx[W]) for W in a.windows]
    if len(a.windows) > 1: feature_sets.append(("FULL + ctx" + "+".join(map(str, a.windows)), FULL + sum((ctx[W] for W in a.windows), [])))
    feature_sets.append(("FULL + visual ctx only " + "+".join(map(str, a.windows)), FULL + sum(([f for f in ctx[W] if f.startswith("p_v")] for W in a.windows), [])))
    CTXW = sum((ctx[W] for W in a.windows), []); THALF = ["t_half_exp20", "t_half_cap600"]; PREV60 = [f"prev60_{c}" for c in range(NUM_CLASSES)]; PREV150 = [f"prev150_{c}" for c in range(NUM_CLASSES)]
    GOALP = [f"prev60_{EVENT_DICTIONARY_V2['Goal']}", f"prev150_{EVENT_DICTIONARY_V2['Goal']}"]
    feature_sets += [("FULL + ctx + t_half", FULL + CTXW + THALF), ("FULL + ctx + t_half + prev goal", FULL + CTXW + THALF + GOALP),
                     ("FULL + ctx + t_half + prev60 all classes", FULL + CTXW + THALF + PREV60), ("FULL + ctx + t_half + prev60 + prev150", FULL + CTXW + THALF + PREV60 + PREV150)]
    if a.sets: feature_sets = [feature_sets[i] for i in a.sets]
    R = {"seed": s, "omega": omega, "windows": a.windows, "systems": {}}
    D_late = [nms_detections(P) for P in Pf_t]; lo, ti = GT.mAP(D_late, "loose"), GT.mAP(D_late, "tight")
    R["systems"]["late fusion"] = {"loose": lo["mAP"], "visible": lo["visible"], "unshown": lo["unshown"], "tight": ti["mAP"], "per_class_unshown": lo["per_class_unshown"]}
    print(f"late fusion: loose {100*lo['mAP']:.1f} vis {100*lo['visible']:.1f} uns {100*lo['unshown']:.1f} tight {100*ti['mAP']:.1f}", flush=True)
    # candidate rows (tau0 set) for the selective-prediction check of the wider features
    rows_v0 = build_candidates(GV, Pf_v, Pv_v, Pa_v, thr=a.tau0, extra=extra_v); rows_t0 = build_candidates(GT, Pf_t, Pv_t, Pa_t, thr=a.tau0, extra=extra_t)
    cache_v, cache_t = {}, {}
    for fname, feats in feature_sets:
        t1 = time.time()
        # reranker: tau_r and combiner chosen by OOF loose mAP on validation (same procedure as the pipeline), using cached rows per thr
        # monkeypatch build_candidates inside select_reranker to include the extra arrays
        import run_pipeline as rp
        orig = rp.build_candidates
        rp.build_candidates = lambda gs, Pf, Pv, Pa, thr=CAND_THR, extra=None: (cache_v if gs is GV else cache_t).setdefault(thr, orig(gs, Pf, Pv, Pa, thr=thr, extra=(extra_v if gs is GV else extra_t)))
        try:
            rcfg = rp.select_reranker(GV, Pf_v, Pv_v, Pa_v, feats, thr_grid=tuple(a.thr_grid), log_fn=lambda m: print("   ", m, flush=True))
            rows_v_r = rp.build_candidates(GV, Pf_v, Pv_v, Pa_v, thr=rcfg["thr"]); rows_t_r = rp.build_candidates(GT, Pf_t, Pv_t, Pa_t, thr=rcfg["thr"])
        finally:
            rp.build_candidates = orig
        sel_r = fit_reliability(rows_v_r, feats, class_cond=True, class_inter=rcfg["class_inter"], C=rcfg["C"]); g_r = sel_r.raw(rows_t_r)
        out = {"reranker": {k: rcfg[k] for k in ("thr", "class_inter", "C", "val_map")}, "n_feat": len(feats)}
        for name, D in (("per half", rerank_score_multiset_per_half(GT, Pf_t, rows_t_r, g_r)), ("test-wide", rerank_score_multiset(GT, Pf_t, rows_t_r, g_r))):
            lo, ti = GT.mAP(D, "loose"), GT.mAP(D, "tight")
            out[name] = {"loose": lo["mAP"], "visible": lo["visible"], "unshown": lo["unshown"], "tight": ti["mAP"], "tight_unshown": ti["unshown"], "per_class_unshown": lo["per_class_unshown"], "per_class": lo["per_class"]}
            print(f"{fname:40s} rerank {name:9s}: loose {100*lo['mAP']:.1f} vis {100*lo['visible']:.1f} uns {100*lo['unshown']:.1f} | tight {100*ti['mAP']:.1f}  "
                  f"IFK {100*lo['per_class_unshown'][EVENT_DICTIONARY_V2['Indirect free-kick']]:.1f} KO {100*lo['per_class_unshown'][EVENT_DICTIONARY_V2['Kick-off']]:.1f} CL {100*lo['per_class_unshown'][EVENT_DICTIONARY_V2['Clearance']]:.1f}", flush=True)
        # selective prediction with the same features (candidate set at tau0): CV-selected combiner, pooled cross-fitted isotonic
        ccfg = select_combiner(rows_v0, feats); est = fit_reliability(rows_v0, feats, class_cond=False, class_inter=ccfg["class_inter"], C=ccfg["C"])
        st = selective_table(rows_t0, est(rows_t0), fname); out["selective"] = {"AURC": st["AURC"], "risk80": st["risk80"], "ECE": st["ECE"], "combiner": {k: ccfg[k] for k in ("class_inter", "C")}}
        print(f"{fname:40s} selective: AURC {st['AURC']:.4f} risk80 {100*st['risk80']:.1f} ECE {st['ECE']:.3f}   ({time.time()-t1:.0f}s)", flush=True)
        R["systems"][fname] = out
        json.dump(R, open(os.path.join(RESULTS, f"unshown_fix_s{s}{a.out_tag}.json"), "w"), indent=2)
    print("done", f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=0); p.add_argument("--windows", type=int, nargs="+", default=[15, 30])
    p.add_argument("--thr-grid", type=float, nargs="+", default=[0.05, 0.1, 0.2]); p.add_argument("--tau0", type=float, default=0.4)
    p.add_argument("--sets", type=int, nargs="*", default=None, help="indices of feature sets to run (default all)"); p.add_argument("--out-tag", default="")
    main(p.parse_args())
