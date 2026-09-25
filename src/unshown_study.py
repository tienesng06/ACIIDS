"""Why does NetVLAD++ (official code, our features) beat our late fusion on UNSHOWN events (37.2 vs 33.1 loose mAP)?
Diagnostics on the test split, seed 0:
  1. per-class unshown AP of both systems (same evaluation code) and the unshown event counts;
  2. for every unshown GT event: distance to the nearest same-class detection and that detection's within-class score
     percentile, for both systems (localisation vs ranking);
  3. cheap remedies on our fused scores: wider NMS windows; temporal max/mean pooling of the fused scores before NMS
     (a score-level analogue of NetVLAD++'s window-level labels), including asymmetric (after-event) pooling.
Writes results/unshown_s0.json.
Usage: python src/unshown_study.py [--seed 0] [--nv-root baselines/eval_stage/netvladpp/run_0]
"""
import os, sys, json, argparse, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from analysis import *
from run_pipeline import available_games, tune_omega

def load_netvlad(root, gs):
    """Official per-game results_spotting.json -> list of (n, C) sparse detection arrays aligned with gs (post-NMS)."""
    out = []
    for g in gs.games:
        p = os.path.join(root, g, "results_spotting.json")
        preds = json.load(open(p))["predictions"] if os.path.exists(p) else []
        for h in (1, 2):
            n = gs.nframes[g][h]; D = np.full((n, NUM_CLASSES), -1.0, dtype=np.float32)
            for a in preds:
                if int(a["gameTime"][0]) != h or a["label"] not in EVENT_DICTIONARY_V2: continue
                f = min(int(FPS * int(a["position"]) / 1000.0), n - 1); c = EVENT_DICTIONARY_V2[a["label"]]
                D[f, c] = max(D[f, c], float(a["confidence"]))
            out.append(D)
    return out

def nearest_detection_stats(gs, dets, which="unshown"):
    """For every GT event of the given visibility: distance (s) to the nearest same-class detection and the within-class
    score percentile of that detection (over all detections of the class in the test split)."""
    pct = {}
    for c in range(NUM_CLASSES):
        s = np.concatenate([D[:, c][D[:, c] >= 0] for D in dets]); pct[c] = np.sort(s)
    rows = []
    k = 0
    for g in gs.games:
        for h in (1, 2):
            D = dets[k]; ev = gs.events[g][h]
            for f, c, vis in ev:
                if (vis == 0) != (which == "unshown"): continue
                idx = np.where(D[:, c] >= 0)[0]
                if len(idx) == 0: rows.append((c, np.inf, 0.0, 0.0)); continue
                j = idx[np.argmin(np.abs(idx - f))]; d = abs(int(j) - f) / FPS; sc = float(D[j, c])
                rows.append((c, d, sc, float(np.searchsorted(pct[c], sc) / max(1, len(pct[c])))))
            k += 1
    return rows

def pool_scores(P, before, after, mode="max"):
    """Temporal pooling of (n, C) scores over [t-before, t+after] frames."""
    n = P.shape[0]; out = np.empty_like(P)
    cs = np.cumsum(np.vstack([np.zeros((1, P.shape[1]), P.dtype), P]), axis=0) if mode == "mean" else None
    for t in range(n):
        lo, hi = max(0, t - before), min(n, t + after + 1)
        out[t] = P[lo:hi].max(axis=0) if mode == "max" else (cs[hi] - cs[lo]) / (hi - lo)
    return out

def main(a):
    s = a.seed
    tags = [f"visual_s{s}", f"audio_s{s}"]
    val_games, test_games = available_games("valid", tags), available_games("test", tags)
    GV, GT = GameSet(val_games), GameSet(test_games)
    Pv_v, Pa_v = [sigmoid(load_logits(tags[0], g, h)) for g in GV.games for h in (1, 2)], [sigmoid(load_logits(tags[1], g, h)) for g in GV.games for h in (1, 2)]
    Pv_t, Pa_t = [sigmoid(load_logits(tags[0], g, h)) for g in GT.games for h in (1, 2)], [sigmoid(load_logits(tags[1], g, h)) for g in GT.games for h in (1, 2)]
    omega = tune_omega(GV, Pv_v, Pa_v); Pf_t = late_fusion(Pv_t, Pa_t, omega); Pf_v = late_fusion(Pv_v, Pa_v, omega)
    print(f"games valid {len(val_games)} test {len(test_games)} omega {omega}", flush=True)
    R = {"seed": s, "omega": omega}

    def ev(dets, name, metric="loose"):
        r = GT.mAP(dets, metric); print(f"{name:42s} {metric}: mAP {100*r['mAP']:.1f} visible {100*r['visible']:.1f} unshown {100*r['unshown']:.1f}", flush=True); return r

    D_late = [nms_detections(P) for P in Pf_t]; D_nv = load_netvlad(a.nv_root, GT)
    D_vis = [nms_detections(P) for P in Pv_t]
    r_late, r_nv, r_vis = ev(D_late, "late fusion (ours)"), ev(D_nv, "NetVLAD++ official run_0"), ev(D_vis, "visual-only (ours)")
    r_late_t, r_nv_t = ev(D_late, "late fusion (ours)", "tight"), ev(D_nv, "NetVLAD++ official run_0", "tight")
    # 1. per-class unshown AP and counts
    n_uns = np.zeros(NUM_CLASSES, int); n_vis = np.zeros(NUM_CLASSES, int)
    for g in GT.games:
        for h in (1, 2):
            for f, c, vis in GT.events[g][h]: (n_vis if vis else n_uns)[c] += 1
    per_class = []
    print("\nper-class unshown AP (loose), n_unshown, n_visible; ours late / NetVLAD++ / ours visual-only")
    for c in np.argsort(-n_uns):
        per_class.append({"cls": CLASSES[c], "n_unshown": int(n_uns[c]), "n_visible": int(n_vis[c]), "late": r_late["per_class_unshown"][c], "netvlad": r_nv["per_class_unshown"][c], "visual": r_vis["per_class_unshown"][c],
                          "late_visible": r_late["per_class_visible"][c], "netvlad_visible": r_nv["per_class_visible"][c]})
        print(f"  {CLASSES[c]:24s} {n_uns[c]:5d} {n_vis[c]:5d}   {100*r_late['per_class_unshown'][c]:5.1f} {100*r_nv['per_class_unshown'][c]:5.1f} {100*r_vis['per_class_unshown'][c]:5.1f}   | visible {100*r_late['per_class_visible'][c]:5.1f} {100*r_nv['per_class_visible'][c]:5.1f}")
    R["per_class"] = per_class
    # 2. nearest detection: localisation vs ranking
    R["nearest"] = {}
    for name, dets in (("late", D_late), ("netvlad", D_nv)):
        for which in ("unshown", "visible"):
            rows = nearest_detection_stats(GT, dets, which); d = np.array([r[1] for r in rows]); p = np.array([r[3] for r in rows])
            st = {"n": len(rows), "within_5s": float(np.mean(d <= 5)), "within_10s": float(np.mean(d <= 10)), "within_30s": float(np.mean(d <= 30)), "within_60s": float(np.mean(d <= 60)),
                  "median_dist_s": float(np.median(d[np.isfinite(d)])), "median_pct_within_60s": float(np.median(p[d <= 60])) if np.any(d <= 60) else None,
                  "mean_pct_within_60s": float(np.mean(p[d <= 60])) if np.any(d <= 60) else None,
                  "hist_dist": {f"{lo}-{hi}": float(np.mean((d > lo) & (d <= hi))) for lo, hi in ((0, 2.5), (2.5, 5), (5, 10), (10, 20), (20, 30), (30, 60))}}
            R["nearest"][f"{name}_{which}"] = st
            print(f"{name:8s} {which:8s} n={st['n']:5d} within 5/10/30/60 s: {st['within_5s']:.2f}/{st['within_10s']:.2f}/{st['within_30s']:.2f}/{st['within_60s']:.2f}  median dist {st['median_dist_s']:.1f} s  median score pct of nearest (<=60 s) {st['median_pct_within_60s']:.2f}")
    # 3. cheap remedies on our fused scores
    R["remedies"] = {}
    print("\nremedies on late fusion (test, seed 0):")
    for win in (20, 40, 60):
        dets = []
        for P in Pf_t:
            D = np.full_like(P, -1.0)
            for c in range(NUM_CLASSES):
                for f, sc in nms_1d(P[:, c], win, thresh=0.0): D[f, c] = sc
            dets.append(D)
        rl, rt = GT.mAP(dets, "loose"), GT.mAP(dets, "tight")
        R["remedies"][f"nms_{win//FPS}s"] = {"loose": rl["mAP"], "visible": rl["visible"], "unshown": rl["unshown"], "tight": rt["mAP"], "tight_unshown": rt["unshown"]}
        print(f"  NMS {win//FPS:2d} s                      loose {100*rl['mAP']:.1f} vis {100*rl['visible']:.1f} uns {100*rl['unshown']:.1f} | tight {100*rt['mAP']:.1f} uns {100*rt['unshown']:.1f}")
    for mode in ("max", "mean"):
        for before, after in ((4, 4), (10, 10), (20, 20), (30, 30), (0, 10), (0, 20), (0, 30), (10, 30)):
            Pp = [pool_scores(P, before, after, mode) for P in Pf_t]; dets = [nms_detections(P) for P in Pp]
            rl, rt = GT.mAP(dets, "loose"), GT.mAP(dets, "tight")
            key = f"pool_{mode}_-{before//FPS}s_+{after//FPS}s"
            R["remedies"][key] = {"loose": rl["mAP"], "visible": rl["visible"], "unshown": rl["unshown"], "tight": rt["mAP"], "tight_unshown": rt["unshown"]}
            print(f"  {key:32s} loose {100*rl['mAP']:.1f} vis {100*rl['visible']:.1f} uns {100*rl['unshown']:.1f} | tight {100*rt['mAP']:.1f} uns {100*rt['unshown']:.1f}", flush=True)
    R["baseline"] = {"late": {"loose": r_late["mAP"], "visible": r_late["visible"], "unshown": r_late["unshown"], "tight": r_late_t["mAP"], "tight_unshown": r_late_t["unshown"]},
                     "netvlad": {"loose": r_nv["mAP"], "visible": r_nv["visible"], "unshown": r_nv["unshown"], "tight": r_nv_t["mAP"], "tight_unshown": r_nv_t["unshown"]},
                     "visual": {"loose": r_vis["mAP"], "visible": r_vis["visible"], "unshown": r_vis["unshown"]}}
    out = os.path.join(RESULTS, f"unshown_s{s}.json"); json.dump(R, open(out, "w"), indent=2); print("written", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=0); p.add_argument("--nv-root", default=os.path.join(ROOT, "baselines", "eval_stage", "netvladpp", "run_0"))
    main(p.parse_args())
