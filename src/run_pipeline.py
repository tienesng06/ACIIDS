"""Full analysis run: all systems, all metrics, CSV outputs + SN-Agree annotations + summary.json.
Assumes models are trained and per-frame logits exist under results/preds/<tag>/<game>/<half>.npy.
Usage: python src/run_pipeline.py --seeds 0 1 2 3 4 [--n-boot 1000] [--fast]
Parallel use: one process per seed, e.g. `--seeds 2 --ens-seeds 0 1 2 3 4 --out-tag _s2`, then `python src/merge_results.py`.
The seed-0 process also exports SN-Agree, the bootstrap CIs and the leave-one-league-out table.
"""
import os, sys, json, csv, argparse, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from analysis import *

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

# feature sets for the reliability estimator (all read from the candidate rows of analysis.build_candidates)
AGREE6 = ["p_v", "p_a", "A_gap", "A_js", "A_prod", "p_f"]                 # branch scores at +-1 s and their agreement (first run)
CTX_V = ["p_v_5s", "p_v_mean5s"]; CTX_A = ["p_a_5s", "p_a_mean5s", "dt_peak"]  # temporal context of each branch over +-5 s
FULL = AGREE6 + CTX_V + CTX_A                                            # ours (11 features)
# event-structure context for the RERANKER only (analysis.context_arrays): wider branch context (mean before / after, max over
# +-15 s and +-30 s), seconds since the half started, and the max fused score of every class over the preceding minute.
# Added after src/unshown_study.py showed that NetVLAD++'s window-level labels score off-camera restarts (kick-off after a
# goal or at the half start, free-kick after a foul) that frame-level branches cannot; selected on validation like the rest.
CTX_WIDE = [f"p_{b}_{k}{W}" for W in CTX_WINDOWS for b in ("v", "a") for k in ("pre", "post", "max")]
STRUCT = ["t_half_exp20", "t_half_cap600"] + [f"prev{PREV_WIN}_{c}" for c in range(NUM_CLASSES)]
FULL_R = FULL + CTX_WIDE + STRUCT   # shared 42-feature representation used by separately selected selector/ranker heads
AGREE11 = FULL                      # the same estimator without event-structure context, kept as one ablation row
NO_AGREEMENT = ["p_v", "p_a", "p_f"] + CTX_V + ["p_a_5s", "p_a_mean5s"] + CTX_WIDE + STRUCT
PAST_LOCAL = ["p_v_past1", "p_a_past1", "p_v_past5", "p_a_past5", "p_v_meanpast5", "p_a_meanpast5", "dt_peak_past", "p_f"]
PAST_WIDE = [f"p_{b}_{k}{W}" for W in CTX_WINDOWS for b in ("v", "a") for k in ("pre", "premax")]
PAST_ONLY = PAST_LOCAL + PAST_WIDE + STRUCT  # available at t; no post-event or symmetric-window features
FEATURE_SETS = [("p_f only", ["p_f"]), ("+A_top", ["p_f", "A_top"]), ("+A_gap", ["p_f", "A_gap"]), ("+A_js", ["p_f", "A_js"]), ("+A_prod", ["p_f", "A_prod"]),
                ("p_f + branch scores + agreement (6)", AGREE6), ("p_f + visual context", ["p_f"] + CTX_V), ("p_f + visual + audio context", ["p_f"] + CTX_V + CTX_A),
                ("all but audio context", AGREE6 + CTX_V), ("agreement + temporal context (11)", FULL),
                ("all (ours, 42: + event-structure context)", FULL_R), ("visual only, no audio anywhere", ["p_v"] + CTX_V)]

def write_csv(path, rows, fields=None):
    if not rows: return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def prior_system_scores(root, gs, prediction_file="results_spotting.json"):
    """Read a prior system's per-game prediction files into (n, C) sparse score arrays aligned with `gs`, so that its
    confidence can be evaluated as a SELECTOR on our candidate set exactly like the PAVE and QMF scores.
    Returns None when the predictions are not present, so the pipeline runs without them."""
    if not os.path.isdir(root): return None
    out = []
    for g in gs.games:
        p = os.path.join(root, g, prediction_file)
        preds = json.load(open(p))["predictions"] if os.path.exists(p) else []
        for h in (1, 2):
            n = gs.nframes[g][h]; A = np.zeros((n, NUM_CLASSES), dtype=np.float32)
            for a in preds:
                if int(a["gameTime"][0]) != h or a["label"] not in EVENT_DICTIONARY_V2: continue
                f = min(int(FPS * int(a["position"]) / 1000.0), n - 1); c = EVENT_DICTIONARY_V2[a["label"]]
                A[f, c] = max(A[f, c], float(a["confidence"]))
            out.append(A)
    return out

def available_games(split, tag_list):
    games = []
    for g in games_for(split):
        if all(os.path.exists(os.path.join(RESULTS, "preds", t, g, "2.npy")) for t in tag_list) and has_game(g, need_audio=False):
            games.append(g)
    return games

def class_priors():
    cnt = np.zeros(NUM_CLASSES)
    for g in games_for("train"):
        p = os.path.join(DATA, g, "Labels-v2.json")
        if not os.path.exists(p): continue
        for a in load_labels(g):
            if a["label"] in EVENT_DICTIONARY_V2: cnt[EVENT_DICTIONARY_V2[a["label"]]] += 1
    return cnt / max(1, cnt.sum()), cnt

def tune_omega(gs, Pv, Pa):
    best = (-1, 0.5)
    for w in np.linspace(0.1, 0.9, 9):
        m = gs.mAP([nms_detections(P) for P in late_fusion(Pv, Pa, w)], "loose")["mAP"]
        if m > best[0]: best = (m, float(w))
    return best[1]

def tune_tau(gs, Zv, Za):
    best = (-1, 1.0)
    for tau in (0.5, 1.0, 2.0, 4.0, 8.0):
        m = gs.mAP([nms_detections(P) for P in energy_weighted_fusion(Zv, Za, tau)], "loose")["mAP"]
        if m > best[0]: best = (m, tau)
    return best[1]

def design(rows, feat_names, class_inter):
    """Feature matrix; with class_inter, appends a class one-hot and feature x class interactions (class-specific
    coefficients, shrunk toward the shared ones by the L2 penalty of the logistic fit)."""
    X = np.array([[r[k] for k in feat_names] for r in rows], dtype=float)
    if not class_inter: return X
    oh = np.eye(NUM_CLASSES)[np.array([r["cls"] for r in rows])]
    return np.hstack([X, oh, np.einsum("ij,ik->ijk", X, oh).reshape(len(X), -1)])


def grouped_splits(rows, n_splits=5):
    """Deterministic match-grouped folds; no match contributes candidates to both sides of a fold."""
    from sklearn.model_selection import GroupKFold
    groups = np.array([r["game"] for r in rows]); n = min(n_splits, len(np.unique(groups)))
    if n < 2: raise ValueError("at least two matches are required for grouped cross-validation")
    return GroupKFold(n_splits=n).split(np.zeros(len(rows)), groups=groups)

def fit_reliability(rows_val, feat_names, class_cond=True, n_min=30, shrink=20.0, class_inter=False, C=1.0, cross_fit=True, n_splits=5):
    """Returns score(rows) -> calibrated reliability r; score.raw(rows) -> the uncalibrated combiner output g(z)
    (tie-free, used for score-multiset-preserving reranking). Single-feature sets are used as they are.
    cross_fit: the isotonic maps are fitted on OUT-OF-MATCH-FOLD combiner scores of the validation candidates (5-fold), not on
    the in-sample scores of a combiner fitted to the same rows; the final combiner is fitted on all validation rows.
    In-sample calibration over-fits small classes (RQ3 study, src/rq3_study.py)."""
    from sklearn.linear_model import LogisticRegression
    y = np.array([r["correct"] for r in rows_val]); c = np.array([r["cls"] for r in rows_val])
    if len(feat_names) == 1 and not class_inter:
        g = None; mu = sd = None; s = design(rows_val, feat_names, False)[:, 0]
    else:
        X = design(rows_val, feat_names, class_inter); mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd
        g = LogisticRegression(max_iter=3000, C=C).fit(Xs, y); s = g.predict_proba(Xs)[:, 1]
        if cross_fit and len(y) >= 10 * n_splits:
            s = np.zeros(len(y))
            for tr, te in grouped_splits(rows_val, n_splits):
                s[te] = LogisticRegression(max_iter=3000, C=C).fit(Xs[tr], y[tr]).predict_proba(Xs[te])[:, 1]
    cal = Reliability(n_min=n_min, shrink=shrink).fit(s, y, c)
    def raw(rows):
        if g is None: return design(rows, feat_names, False)[:, 0]
        return g.predict_proba((design(rows, feat_names, class_inter) - mu) / sd)[:, 1]
    def score(rows):
        ct = np.array([r["cls"] for r in rows]); return cal.predict(raw(rows), ct, class_conditional=class_cond)
    score.raw = raw
    return score

def select_combiner(rows_v, feat_names, grid_C=(0.03, 0.1, 0.3, 1.0), n_splits=5, log_fn=None):
    """Choose (class_inter, C) by 5-fold match-grouped AURC on the validation candidates (calibrator fitted inside
    each fold). Nothing here touches the test split."""
    y = np.array([r["correct"] for r in rows_v]); best = None; table = {}
    for ci in (False, True):
        for C in grid_C:
            oof = np.zeros(len(rows_v))
            for tr, te in grouped_splits(rows_v, n_splits):
                sc = fit_reliability([rows_v[i] for i in tr], feat_names, True, class_inter=ci, C=C); oof[te] = sc([rows_v[i] for i in te])
            a = risk_coverage(oof, y)[2]; table[f"class_inter={ci},C={C}"] = a
            if best is None or a < best[0]: best = (a, ci, C)
    if log_fn: log_fn(f"combiner selection (5-fold match-grouped CV AURC on validation): best class_inter={best[1]} C={best[2]} AURC={best[0]:.4f}")
    return {"class_inter": best[1], "C": best[2], "cv_aurc": best[0], "table": table}

def select_reranker(gs_v, Pf_v, Pv_v, Pa_v, feat_names, thr_grid=(0.05, 0.1, 0.2), grid_C=(0.1, 0.3, 1.0), n_splits=5, log_fn=None):
    """Choose the reranking set threshold tau_r and the combiner (class_inter, C) by out-of-fold loose mAP on the
    validation split (5-fold by match, score-multiset-preserving reranking with the OOF combiner output)."""
    best = None; table = {}
    for thr in thr_grid:
        rows = build_candidates(gs_v, Pf_v, Pv_v, Pa_v, thr=thr)
        for ci in (False, True):
            for C in grid_C:
                oof = np.zeros(len(rows))
                for tr, te in grouped_splits(rows, n_splits):
                    sc = fit_reliability([rows[i] for i in tr], feat_names, True, class_inter=ci, C=C, cross_fit=False); oof[te] = sc.raw([rows[i] for i in te])
                m = gs_v.mAP(rerank_score_multiset(gs_v, Pf_v, rows, oof), "loose")["mAP"]; table[f"thr={thr},class_inter={ci},C={C}"] = m
                if best is None or m > best[0]: best = (m, thr, ci, C)
    if log_fn: log_fn(f"reranker selection (OOF loose mAP on validation): best thr={best[1]} class_inter={best[2]} C={best[3]} mAP={best[0]:.4f}")
    return {"thr": best[1], "class_inter": best[2], "C": best[3], "val_map": best[0], "table": table}

def apply_rerank(gs, Pf, rows, r, lam):
    """First (flawed) reranker: candidates get s = p_f * r^lam, every other detection keeps p_f, so re-scored candidates can fall below tau0 and rank under untouched detections (kept as a documented row)."""
    out = [np.zeros_like(P) for P in Pf]; idx = {}
    k = 0
    for g in gs.games:
        for h in (1, 2): idx[(g, h)] = k; k += 1
    for P, D in zip(Pf, out):
        D[:] = nms_detections(P)
    for row, rr in zip(rows, r):
        D = out[idx[(row["game"], row["half"])]]; f, c = row["frame"], row["cls"]
        D[f, c] = row["p_f"] * (max(rr, 1e-3) ** lam)
    return out  # entries stay -1 where there is no detection

def selective_table(rows, scores, name):
    y = np.array([r["correct"] for r in rows]); cov, risk, au = risk_coverage(scores, y)
    alo, ahi = risk_coverage_tie_bounds(scores, y)
    return {"selector": name, "AURC": au, "risk80": risk_at_coverage(scores, y, 0.8), "risk90": risk_at_coverage(scores, y, 0.9),
            "AURC_tie_low": alo, "AURC_tie_high": ahi, "ECE": ece(scores, y), "Brier": brier(scores, y),
            "spearman": spearman(scores, y)[0], "n": len(rows)}

def attach_detection_scores(rows, gs, detections, key):
    """Attach the score at each candidate location from an aligned detection list."""
    idx = {}; k = 0
    for game in gs.games:
        for half in (1, 2): idx[(game, half)] = k; k += 1
    for row in rows:
        i = idx[(row["game"], row["half"])]
        row[key] = float(detections[i][row["frame"], row["cls"]])

def main(args):
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ensure_dirs(); rng = np.random.RandomState(0)
    seeds = args.seeds; nB = args.n_boot; all_seeds = sorted(set(args.ens_seeds) | set(seeds)); tag = args.out_tag
    OUT = os.path.join(RESULTS, args.out_dir) if args.out_dir else RESULTS; os.makedirs(OUT, exist_ok=True)
    prior, counts = class_priors(); rare = [c for c in range(NUM_CLASSES) if prior[c] < RARE_PRIOR]
    log("rare classes:", [CLASSES[c] for c in rare])
    tags_needed = [f"visual_s{s}" for s in all_seeds] + [f"audio_s{s}" for s in seeds] + [f"joint_s{s}" for s in seeds] + [f"qmf_s{s}" for s in seeds]
    val_games, test_games = available_games("valid", tags_needed), available_games("test", tags_needed)
    if args.fast: val_games, test_games = val_games[:20], test_games[:20]
    log(f"valid games {len(val_games)} test games {len(test_games)}")
    GV, GT = GameSet(val_games), GameSet(test_games)
    main_rows, sel_rows, diag_rows, abl_rows, rare_rows, deg_rows = [], [], [], [], [], []
    per_seed = {}; tau0_by_seed, omega_by_seed, tau_by_seed, lambda_by_seed, combiner_by_seed, rerank_by_seed, rare_val_by_seed = {}, {}, {}, {}, {}, {}, {}
    for s in seeds:
        log(f"=== seed {s}")
        Zv_v, Za_v = [load_logits(f"visual_s{s}", g, h) for g in GV.games for h in (1, 2)], [load_logits(f"audio_s{s}", g, h) for g in GV.games for h in (1, 2)]
        Zv_t, Za_t = [load_logits(f"visual_s{s}", g, h) for g in GT.games for h in (1, 2)], [load_logits(f"audio_s{s}", g, h) for g in GT.games for h in (1, 2)]
        Pv_v, Pa_v, Pv_t, Pa_t = [sigmoid(z) for z in Zv_v], [sigmoid(z) for z in Za_v], [sigmoid(z) for z in Zv_t], [sigmoid(z) for z in Za_t]
        Pj_t = [sigmoid(load_logits(f"joint_s{s}", g, h)) for g in GT.games for h in (1, 2)]
        omega = tune_omega(GV, Pv_v, Pa_v); tau = float("nan")
        Pf_v, Pf_t = late_fusion(Pv_v, Pa_v, omega), late_fusion(Pv_t, Pa_t, omega)
        # real QMF (Zhang et al. 2023; official objective ported in train_branch.QMFNet): fused softmax probabilities of the trained model
        Pq_t = [softmax_events(load_logits(f"qmf_s{s}", g, h)) for g in GT.games for h in (1, 2)]; Pq_v = [softmax_events(load_logits(f"qmf_s{s}", g, h)) for g in GV.games for h in (1, 2)]
        log(f"omega={omega} tau={tau}"); omega_by_seed[s] = omega; tau_by_seed[s] = tau
        # same-type ensembles (4 visual seeds, excluding s if possible)
        ens_seeds = [x for x in args.ens_seeds if x != s][:4] if len(args.ens_seeds) > 4 else list(args.ens_seeds)[:4]
        Pens_t = [[sigmoid(load_logits(f"visual_s{e}", g, h)) for g in GT.games for h in (1, 2)] for e in ens_seeds]
        D_mean = [nms_detections(P) for P in mean_ensemble(Pens_t)]
        D_members = [[nms_detections(P) for P in Pm] for Pm in Pens_t]
        D_agree = [agreement_ensemble([D_members[m][i] for m in range(len(ens_seeds))]) for i in range(len(D_mean))]
        D_agree_byp = [agreement_ensemble([D_members[m][i] for m in range(len(ens_seeds))], bypass_classes=tuple(rare)) for i in range(len(D_mean))]
        systems = {"Visual-only": [nms_detections(P) for P in Pv_t], "Audio-only": [nms_detections(P) for P in Pa_t],
                   "Late fusion (fixed w)": [nms_detections(P) for P in Pf_t], "Early fusion (joint)": [nms_detections(P) for P in Pj_t],
                   "Mean ensemble (4 visual seeds)": D_mean, "Agreement ensemble (PAVE-style)": D_agree, "Agreement ensemble + rare bypass": D_agree_byp,
                   "QMF (Zhang et al.)": [nms_detections(P) for P in Pq_t]}
        # candidates: operating point = F1-optimal threshold on validation
        tau0, f1v = f1_optimal_threshold(GV, Pf_v); log(f"tau0 (F1-optimal on validation) = {tau0} (F1 {f1v:.3f})"); tau0_by_seed[s] = tau0
        rows_v = build_candidates(GV, Pf_v, Pv_v, Pa_v, thr=tau0); rows_t = build_candidates(GT, Pf_t, Pv_t, Pa_t, thr=tau0)
        # attach ensemble-agreement and QMF confidence to test/val candidates
        def attach(rows, gs, Dag, Pq, priors=()):
            idx = {}; k = 0
            for g in gs.games:
                for h in (1, 2): idx[(g, h)] = k; k += 1
            for r in rows:
                i = idx[(r["game"], r["half"])]; f, c = r["frame"], r["cls"]
                lo, hi = max(0, f - 3), min(Dag[i].shape[0], f + 4)
                r["ens_agree"] = float(Dag[i][lo:hi, c].max()); r["p_qmf"] = float(Pq[i][lo:hi, c].max())
                # a prior system's confidence for the same class near the same time (0 where it emitted nothing);
                # the window is the +-CORR_DELTA correctness window, since these systems localise less tightly
                for name, arr in priors:
                    lo2, hi2 = max(0, f - CORR_DELTA), min(arr[i].shape[0], f + CORR_DELTA + 1)
                    r[name] = float(arr[i][lo2:hi2, c].max())
        D_agree_v = None
        Pens_v = [[sigmoid(load_logits(f"visual_s{e}", g, h)) for g in GV.games for h in (1, 2)] for e in ens_seeds]
        D_members_v = [[nms_detections(P) for P in Pm] for Pm in Pens_v]
        D_agree_v = [agreement_ensemble([D_members_v[m][i] for m in range(len(ens_seeds))]) for i in range(len(Pf_v))]
        # prior systems as selectors on the same candidates (seed-matched run where available, else run_0)
        PRIOR_ROOTS = [("p_netvlad", os.path.join(ROOT, "baselines", "eval_stage", "netvladpp"), "results_spotting.json"),
                       ("p_calf", os.path.join(ROOT, "baselines", "eval_stage", "calf"), "Predictions-v2.json")]
        priors_t, priors_v = [], []
        for nm, rt, pf in PRIOR_ROOTS:
            d = os.path.join(rt, f"run_{s}")
            if not os.path.isdir(d): d = os.path.join(rt, "run_0")
            at = prior_system_scores(d, GT, pf)
            if at is not None: priors_t.append((nm, at)); log(f"prior selector {nm}: {os.path.basename(d)}")
        attach(rows_v, GV, D_agree_v, Pq_v); attach(rows_t, GT, D_agree, Pq_t, priors_t)
        log(f"candidates valid {len(rows_v)} test {len(rows_t)}")
        cfg = select_combiner(rows_v, FULL_R, log_fn=log); CI, CC = cfg["class_inter"], cfg["C"]; combiner_by_seed[s] = {"class_inter": CI, "C": CC, "cv_aurc": cfg["cv_aurc"]}
        sel_conf = fit_reliability(rows_v, ["p_f"], class_cond=False); sel_pool = fit_reliability(rows_v, FULL_R, class_cond=False, class_inter=CI, C=CC)
        sel_conf_cc = fit_reliability(rows_v, ["p_f"], class_cond=True)   # calibration-only control: class-conditional map of p_f
        sel_full = fit_reliability(rows_v, FULL_R, class_cond=False, class_inter=CI, C=CC)    # AGREE: pooled cross-fitted isotonic on the class-specific combiner
        sel_full_cc = fit_reliability(rows_v, FULL_R, class_cond=True, class_inter=CI, C=CC)  # class-conditional calibration stage: ablation
        cfg11 = select_combiner(rows_v, AGREE11, log_fn=log)                                  # ablation: same estimator without event-structure context
        sel_full_11 = fit_reliability(rows_v, AGREE11, class_cond=False, class_inter=cfg11["class_inter"], C=cfg11["C"])
        cfg_noagree = select_combiner(rows_v, NO_AGREEMENT, log_fn=log)
        sel_noagree = fit_reliability(rows_v, NO_AGREEMENT, class_cond=False, class_inter=cfg_noagree["class_inter"], C=cfg_noagree["C"])
        cfg_past = select_combiner(rows_v, PAST_ONLY, log_fn=log)
        sel_past = fit_reliability(rows_v, PAST_ONLY, class_cond=False, class_inter=cfg_past["class_inter"], C=cfg_past["C"])
        sel_ens = fit_reliability(rows_v, ["ens_agree"], class_cond=False)
        sel_qmf = fit_reliability(rows_v, ["p_qmf"], class_cond=False)
        r_t = sel_full(rows_t); r_v = sel_full(rows_v); g_t = sel_full.raw(rows_t)
        # reranking (ours): score-multiset-preserving reordering of ALL detections >= tau_r by a combiner fitted on validation
        # detections >= tau_r; tau_r and the combiner are chosen on validation (select_reranker). The score-compressing
        # variant s = p_f r^lambda of the first run is kept as a system row to document the artifact it produces in the
        # threshold-sampled official AP.
        rcfg = select_reranker(GV, Pf_v, Pv_v, Pa_v, FULL_R, log_fn=log); rerank_by_seed[s] = {k: rcfg[k] for k in ("thr", "class_inter", "C", "val_map")}
        rows_v_r = build_candidates(GV, Pf_v, Pv_v, Pa_v, thr=rcfg["thr"]); rows_t_r = build_candidates(GT, Pf_t, Pv_t, Pa_t, thr=rcfg["thr"])
        sel_rerank = fit_reliability(rows_v_r, FULL_R, class_cond=True, class_inter=rcfg["class_inter"], C=rcfg["C"], cross_fit=False)
        # ablation: the same reranker with the 11 reliability features only (no event-structure context), selected the same way
        rcfg11 = select_reranker(GV, Pf_v, Pv_v, Pa_v, FULL, log_fn=log); rerank_by_seed[s]["no_struct"] = {k: rcfg11[k] for k in ("thr", "class_inter", "C", "val_map")}
        rows_v_r11 = build_candidates(GV, Pf_v, Pv_v, Pa_v, thr=rcfg11["thr"]); rows_t_r11 = build_candidates(GT, Pf_t, Pv_t, Pa_t, thr=rcfg11["thr"])
        g_r11 = fit_reliability(rows_v_r11, FULL, class_cond=True, class_inter=rcfg11["class_inter"], C=rcfg11["C"], cross_fit=False).raw(rows_t_r11)
        rcfg_noagree = select_reranker(GV, Pf_v, Pv_v, Pa_v, NO_AGREEMENT, log_fn=log)
        rows_v_rna = build_candidates(GV, Pf_v, Pv_v, Pa_v, thr=rcfg_noagree["thr"]); rows_t_rna = build_candidates(GT, Pf_t, Pv_t, Pa_t, thr=rcfg_noagree["thr"])
        g_rna = fit_reliability(rows_v_rna, NO_AGREEMENT, class_cond=True, class_inter=rcfg_noagree["class_inter"], C=rcfg_noagree["C"], cross_fit=False).raw(rows_t_rna)
        rcfg_past = select_reranker(GV, Pf_v, Pv_v, Pa_v, PAST_ONLY, log_fn=log)
        rows_v_rpast = build_candidates(GV, Pf_v, Pv_v, Pa_v, thr=rcfg_past["thr"]); rows_t_rpast = build_candidates(GT, Pf_t, Pv_t, Pa_t, thr=rcfg_past["thr"])
        g_rpast = fit_reliability(rows_v_rpast, PAST_ONLY, class_cond=True, class_inter=rcfg_past["class_inter"], C=rcfg_past["C"], cross_fit=False).raw(rows_t_rpast)
        lam = 0.5; lambda_by_seed[s] = lam
        g_r = sel_rerank.raw(rows_t_r)
        systems["AGREE rerank (ours)"] = rerank_score_multiset_per_half(GT, Pf_t, rows_t_r, g_r)          # deployable: multiset reassigned within each half
        systems["AGREE rerank, test-set-wide multiset"] = rerank_score_multiset(GT, Pf_t, rows_t_r, g_r)  # transductive variant

        # Selective diagnostic for the transductive variant. Validation scores use match-grouped out-of-fold
        # ranker outputs before test-set-wide reassignment. Isotonic calibration sees validation labels only.
        g_r_v_oof = np.zeros(len(rows_v_r))
        for tr, te in grouped_splits(rows_v_r):
            fold = fit_reliability([rows_v_r[i] for i in tr], FULL_R, class_cond=True,
                                   class_inter=rcfg["class_inter"], C=rcfg["C"], cross_fit=False)
            g_r_v_oof[te] = fold.raw([rows_v_r[i] for i in te])
        D_tw_v = rerank_score_multiset(GV, Pf_v, rows_v_r, g_r_v_oof)
        attach_detection_scores(rows_v, GV, D_tw_v, "p_transductive")
        attach_detection_scores(rows_t, GT, systems["AGREE rerank, test-set-wide multiset"], "p_transductive")
        sel_transductive = fit_reliability(rows_v, ["p_transductive"], class_cond=False)
        r_transductive = sel_transductive(rows_t)
        systems["AGREE rerank, per half, 11 features (no event-structure context)"] = rerank_score_multiset_per_half(GT, Pf_t, rows_t_r11, g_r11)
        systems["AGREE rerank, per half, no agreement variables"] = rerank_score_multiset_per_half(GT, Pf_t, rows_t_rna, g_rna)
        systems["AGREE rerank, per half, past-only context"] = rerank_score_multiset_per_half(GT, Pf_t, rows_t_rpast, g_rpast)
        systems["AGREE rerank, score-compressing (p_f r^0.5)"] = apply_rerank(GT, Pf_t, rows_t, r_t, lam)
        # ---- main table
        res = {}
        for name, D in systems.items():
            lo, ti = GT.mAP(D, "loose"), GT.mAP(D, "tight"); res[name] = (lo, ti, D)
            main_rows.append({"seed": s, "system": name, "loose": lo["mAP"], "tight": ti["mAP"], "visible": lo["visible"], "unshown": lo["unshown"], **{f"pc_{CLASSES[c]}": lo["per_class"][c] for c in range(NUM_CLASSES)}})
        # ---- selective table
        y_t = np.array([r["correct"] for r in rows_t])
        selectors = {"Confidence-only (calibrated p_f)": sel_conf(rows_t), "Same-type ensemble agreement": sel_ens(rows_t), "QMF fused confidence": sel_qmf(rows_t),
                     "Confidence-only, class-conditional": sel_conf_cc(rows_t), "AGREE without event-structure context": sel_full_11(rows_t),
                     "AGREE, pooled calibration": r_t, "AGREE, class-conditional (ours)": sel_full_cc(rows_t),
                     "AGREE, transductive score reassignment": r_transductive}
        for nm, label in (("p_netvlad", "NetVLAD++ confidence"), ("p_calf", "CALF confidence")):
            if rows_t and nm in rows_t[0]: selectors[label] = np.array([r[nm] for r in rows_t])
        for name, sc in selectors.items(): sel_rows.append({"seed": s, **selective_table(rows_t, sc, name)})
        # ---- diagnostic table: Spearman per agreement score, overall / near a visible event / near an unshown event
        # (visibility of the nearest same-class event within 60 s, defined for correct and wrong candidates alike); error correlation
        for feat in ["A_top", "A_gap", "A_js", "A_prod", "p_v", "p_a", "p_f"]:
            x = np.array([r[feat] for r in rows_t])
            vis = np.array([r["vis_near"] for r in rows_t])
            diag_rows.append({"seed": s, "feature": feat, "spearman_all": spearman(x, y_t)[0],
                              "spearman_visible": spearman(x[vis != 0], y_t[vis != 0])[0], "spearman_unshown": spearman(x[vis == 0], y_t[vis == 0])[0]})
        ev = np.array([int(r["p_v"] >= 0.5) != r["correct"] for r in rows_t]); ea = np.array([int(r["p_a"] >= 0.5) != r["correct"] for r in rows_t])
        e2 = np.array([int(r["ens_agree"] > 0) != r["correct"] for r in rows_t])
        # error correlation between branches: phi coefficient of error indicators; ensemble: visual seed s vs another seed
        other = ens_seeds[0]; Pv2_t = [sigmoid(load_logits(f"visual_s{other}", g, h)) for g in GT.games for h in (1, 2)]
        idx = {}; k = 0
        for g in GT.games:
            for h in (1, 2): idx[(g, h)] = k; k += 1
        ev2 = np.array([int(local_max(Pv2_t[idx[(r["game"], r["half"])]], r["frame"], r["cls"]) >= 0.5) != r["correct"] for r in rows_t])
        # candidate-set phi is a selection artifact (candidates are fused detections >= tau0, which anti-correlates the branches);
        # kept in the CSV under 'errcorr_cand_*' for transparency but NOT reported in the paper.
        diag_rows.append({"seed": s, "feature": "errcorr_cand_visual_audio", "spearman_all": phi(ev, ea), "spearman_visible": float("nan"), "spearman_unshown": float("nan")})
        diag_rows.append({"seed": s, "feature": "errcorr_cand_visual_visual", "spearman_all": phi(ev, ev2), "spearman_visible": float("nan"), "spearman_unshown": float("nan")})
        # error correlation on an UNSELECTED set: all GT events + equal number of random negatives, per-branch decision at 0.5 within +-5 s
        items = unselected_error_set(GT, seed=s); Ev, Ea, Ev2 = branch_errors(Pv_t, items), branch_errors(Pa_t, items), branch_errors(Pv2_t, items)
        yu = np.array([it[3] for it in items], float)
        for nm, e in [("visual", Ev), ("audio", Ea), ("visual_seed", Ev2)]:
            diag_rows.append({"seed": s, "feature": f"errrate_unsel_{nm}", "spearman_all": float(e.mean()), "spearman_visible": float(e[yu == 1].mean()), "spearman_unshown": float(e[yu == 0].mean())})
        diag_rows.append({"seed": s, "feature": "errcorr_unsel_visual_audio", "spearman_all": phi(Ev, Ea), "spearman_visible": phi(Ev[yu == 1], Ea[yu == 1]), "spearman_unshown": phi(Ev[yu == 0], Ea[yu == 0])})
        diag_rows.append({"seed": s, "feature": "errcorr_unsel_visual_visual", "spearman_all": phi(Ev, Ev2), "spearman_visible": phi(Ev[yu == 1], Ev2[yu == 1]), "spearman_unshown": phi(Ev[yu == 0], Ev2[yu == 0])})
        diag_rows.append({"seed": s, "feature": "n_unsel", "spearman_all": float(len(items)), "spearman_visible": float((yu == 1).sum()), "spearman_unshown": float((yu == 0).sum())})
        # reliability bins (deciles of A_prod) overall / near visible / near unshown
        ap = np.array([r["A_prod"] for r in rows_t]); vis = np.array([r["vis_near"] for r in rows_t]); qs = np.quantile(ap, np.linspace(0, 1, 11))
        for b in range(10):
            m = (ap >= qs[b]) & (ap <= qs[b + 1]) if b == 9 else (ap >= qs[b]) & (ap < qs[b + 1])
            diag_rows.append({"seed": s, "feature": f"bin{b}", "spearman_all": float(y_t[m].mean()) if m.any() else float("nan"),
                              "spearman_visible": float(y_t[m & (vis != 0)].mean()) if (m & (vis != 0)).any() else float("nan"),
                              "spearman_unshown": float(y_t[m & (vis == 0)].mean()) if (m & (vis == 0)).any() else float("nan")})
        # ---- rare-class table (recall of correct rare candidates kept at 80% coverage) + per-class ECE
        def rare_recall(sc, cov=0.8):
            keep = acceptance_probability(sc, cov)
            m = np.array([r["cls"] in rare for r in rows_t]) & (y_t == 1)
            return float(keep[m].mean()) if m.any() else float("nan"), int(m.sum())
        for name, sc in [("Confidence-only", selectors["Confidence-only (calibrated p_f)"]), ("Confidence-only class-conditional", selectors["Confidence-only, class-conditional"]), ("AGREE pooled", r_t), ("AGREE class-conditional", selectors["AGREE, class-conditional (ours)"])]:
            rr, n_r = rare_recall(sc); rr90, _ = rare_recall(sc, 0.9)
            cls_arr = np.array([r["cls"] for r in rows_t]); rare_mask = np.isin(cls_arr, rare)
            rare_rows.append({"seed": s, "selector": name, "rare_recall80": rr, "rare_recall90": rr90, "n_rare_correct": n_r,
                              "rare_ece": ece(sc[rare_mask], y_t[rare_mask]) if rare_mask.any() else float("nan"),
                              "rare_mean_r": float(sc[rare_mask].mean()) if rare_mask.any() else float("nan"), "rare_precision": float(y_t[rare_mask].mean()) if rare_mask.any() else float("nan")})
        # ---- ablations: agreement feature sets, lambda, tau0
        for fname, fl in FEATURE_SETS:   # shared-coefficient logistic (C=1), pooled calibration, so the rows differ only in features
            sc = fit_reliability(rows_v, fl, class_cond=False)(rows_t); t = selective_table(rows_t, sc, fname)
            abl_rows.append({"seed": s, "ablation": "features", "setting": fname, "AURC": t["AURC"], "risk80": t["risk80"], "loose": float("nan")})
        t = selective_table(rows_t, r_t, "ours"); abl_rows.append({"seed": s, "ablation": "features", "setting": "all 42 + selected coefficients (ours)", "AURC": t["AURC"], "risk80": t["risk80"], "loose": float("nan")})
        t = selective_table(rows_t, sel_noagree(rows_t), "no-agreement"); abl_rows.append({"seed": s, "ablation": "features", "setting": "all except agreement variables, selected", "AURC": t["AURC"], "risk80": t["risk80"], "loose": float("nan")})
        t = selective_table(rows_t, sel_past(rows_t), "past-only"); abl_rows.append({"seed": s, "ablation": "features", "setting": "past-only context, selected", "AURC": t["AURC"], "risk80": t["risk80"], "loose": float("nan")})
        t = selective_table(rows_t, sel_full_cc(rows_t), "cc"); abl_rows.append({"seed": s, "ablation": "features", "setting": "ours + class-conditional calibration, Eq. (3)", "AURC": t["AURC"], "risk80": t["risk80"], "loose": float("nan")})
        rare_val = {CLASSES[c]: int(sum(1 for r in rows_v if r["cls"] == c)) for c in rare}; rare_val_by_seed[s] = rare_val
        # reranking variants (loose mAP)
        g6_t = fit_reliability(rows_v, AGREE6, class_cond=True, cross_fit=False).raw(rows_t)
        for nm, D in [("late fusion (no reranking)", systems["Late fusion (fixed w)"]), ("score-multiset-preserving, candidates >= tau0, 6 features", rerank_score_multiset(GT, Pf_t, rows_t, g6_t)),
                      ("score-multiset-preserving, candidates >= tau0, 11 features", rerank_score_multiset(GT, Pf_t, rows_t, g_t)),
                      ("score-multiset-preserving, all detections >= tau_r, per half, 11 features", systems["AGREE rerank, per half, 11 features (no event-structure context)"]),
                      ("score-multiset-preserving, per half, no agreement variables", systems["AGREE rerank, per half, no agreement variables"]),
                      ("score-multiset-preserving, per half, past-only context", systems["AGREE rerank, per half, past-only context"]),
                      ("score-multiset-preserving, all detections >= tau_r, per half, + event-structure context (ours)", systems["AGREE rerank (ours)"]), ("score-multiset-preserving, all detections >= tau_r, test-set-wide", systems["AGREE rerank, test-set-wide multiset"]), ("score-compressing p_f r^0.5", systems["AGREE rerank, score-compressing (p_f r^0.5)"])]:
            abl_rows.append({"seed": s, "ablation": "rerank", "setting": nm, "AURC": float("nan"), "risk80": float("nan"), "loose": GT.mAP(D, "loose")["mAP"]})
        for thr in (0.05, 0.1, 0.2):
            rv2 = build_candidates(GV, Pf_v, Pv_v, Pa_v, thr=thr); rt2 = build_candidates(GT, Pf_t, Pv_t, Pa_t, thr=thr)
            sc = fit_reliability(rv2, FULL_R, class_cond=True)(rt2); sc0 = fit_reliability(rv2, ["p_f"], class_cond=False)(rt2)
            ta, tc = selective_table(rt2, sc, "a"), selective_table(rt2, sc0, "c")
            abl_rows.append({"seed": s, "ablation": "tau0", "setting": str(thr), "AURC": ta["AURC"], "risk80": ta["risk80"], "loose": float("nan"), "AURC_conf": tc["AURC"]})
        # ---- degradation conditions (if predictions exist)
        for cond in ["vdrop0.25", "vdrop0.5", "vdrop0.75", "amute0.5"]:
            tv = f"visual_s{s}" if cond.startswith("amute") else f"visual_s{s}_{cond}"
            ta_ = f"audio_s{s}_{cond}" if cond.startswith("amute") else f"audio_s{s}"
            tq = f"qmf_s{s}_{cond}"
            if not all(os.path.exists(os.path.join(RESULTS, "preds", t)) for t in (tv, ta_, tq)): continue
            Zv_d = [load_logits(tv, g, h) for g in GT.games for h in (1, 2)]; Za_d = [load_logits(ta_, g, h) for g in GT.games for h in (1, 2)]
            Pv_d, Pa_d = [sigmoid(z) for z in Zv_d], [sigmoid(z) for z in Za_d]; Pf_d = late_fusion(Pv_d, Pa_d, omega); Pq_d = [softmax_events(load_logits(tq, g, h)) for g in GT.games for h in (1, 2)]
            rows_d = build_candidates(GT, Pf_d, Pv_d, Pa_d, thr=tau0); r_d = sel_full(rows_d); y_d = np.array([r["correct"] for r in rows_d])
            rows_d_r = build_candidates(GT, Pf_d, Pv_d, Pa_d, thr=rcfg["thr"])
            deg_rows.append({"seed": s, "condition": cond, "loose_late": GT.mAP([nms_detections(P) for P in Pf_d], "loose")["mAP"],
                             "loose_qmf": GT.mAP([nms_detections(P) for P in Pq_d], "loose")["mAP"],
                             "loose_agree": GT.mAP(rerank_score_multiset_per_half(GT, Pf_d, rows_d_r, sel_rerank.raw(rows_d_r)), "loose")["mAP"],
                             "AURC_conf": risk_coverage(sel_conf(rows_d), y_d)[2], "AURC_agree": risk_coverage(r_d, y_d)[2], "n_cand": len(rows_d)})
        # ---- SN-Agree export (seed 0 only) and bootstrap CIs
        if s == args.export_seed:
            r_cc_t = sel_full_cc(rows_t)
            for r, rr, rc in zip(rows_t, r_t, r_cc_t): r["r_agree"] = float(rr); r["r_agree_cc"] = float(rc)
            sc_conf = selectors["Confidence-only (calibrated p_f)"]; sc_cc = selectors["Confidence-only, class-conditional"]
            for i, r in enumerate(rows_t):
                r["r_conf"] = float(sc_conf[i]); r["r_conf_cc"] = float(sc_cc[i])
                r["r_transductive"] = float(r_transductive[i]); r["class_name"] = CLASSES[r["cls"]]
            for i, r in enumerate(rows_v): r["r_agree"] = float(r_v[i]); r["class_name"] = CLASSES[r["cls"]]
            write_csv(os.path.join(OUT, f"sn_agree_test{tag}.csv"), rows_t); write_csv(os.path.join(OUT, f"sn_agree_valid{tag}.csv"), rows_v)
            # AURC CI: paired bootstrap over games on the difference conf - agree
            def aurc_diff(sub):
                y = np.array([r["correct"] for r in sub]); a = risk_coverage(np.array([r["r_agree"] for r in sub]), y)[2]; c = risk_coverage(np.array([r["r_conf"] for r in sub]), y)[2]; return c - a
            def aurc_rel(sub):
                y = np.array([r["correct"] for r in sub]); a = risk_coverage(np.array([r["r_agree"] for r in sub]), y)[2]; c = risk_coverage(np.array([r["r_conf"] for r in sub]), y)[2]; return 100 * (c - a) / c
            p_d, lo_d, hi_d = bootstrap_by_game(rows_t, aurc_diff, n_boot=nB); p_r, lo_r, hi_r = bootstrap_by_game(rows_t, aurc_rel, n_boot=nB)
            # decomposition CI: class-conditional confidence-only (calibration alone) vs full AGREE (calibration + agreement)
            def aurc_diff_cc(sub):
                y = np.array([r["correct"] for r in sub]); a = risk_coverage(np.array([r["r_agree"] for r in sub]), y)[2]; c = risk_coverage(np.array([r["r_conf_cc"] for r in sub]), y)[2]; return c - a
            def aurc_rel_cc(sub):
                y = np.array([r["correct"] for r in sub]); a = risk_coverage(np.array([r["r_agree"] for r in sub]), y)[2]; c = risk_coverage(np.array([r["r_conf_cc"] for r in sub]), y)[2]; return 100 * (c - a) / c
            def aurc_diff_cal(sub):   # pooled confidence-only vs class-conditional confidence-only (the calibration part)
                y = np.array([r["correct"] for r in sub]); a = risk_coverage(np.array([r["r_conf_cc"] for r in sub]), y)[2]; c = risk_coverage(np.array([r["r_conf"] for r in sub]), y)[2]; return c - a
            p_dc, lo_dc, hi_dc = bootstrap_by_game(rows_t, aurc_diff_cc, n_boot=nB); p_rc, lo_rc, hi_rc = bootstrap_by_game(rows_t, aurc_rel_cc, n_boot=nB)
            p_cal, lo_cal, hi_cal = bootstrap_by_game(rows_t, aurc_diff_cal, n_boot=nB)
            # mAP CI: paired match bootstrap. Per-match matching is cached, which makes 1,000 official-metric resamples practical.
            nb_map = min(nB, args.n_boot_map); D_late, D_ag, D_tw = systems["Late fusion (fixed w)"], systems["AGREE rerank (ours)"], systems["AGREE rerank, test-set-wide multiset"]
            D_ns = systems["AGREE rerank, per half, 11 features (no event-structure context)"]
            picks = rng.randint(0, len(GT.games), size=(nb_map, len(GT.games)))
            b_late = bootstrap_maps_fast(GT, D_late, picks); b_ag = bootstrap_maps_fast(GT, D_ag, picks)
            b_tw = bootstrap_maps_fast(GT, D_tw, picks); b_ns = bootstrap_maps_fast(GT, D_ns, picks)
            diffs = b_ag[0] - b_late[0]; diffs_tw = b_tw[0] - b_late[0]
            diffs_ctx = b_ag[0] - b_ns[0]; diffs_uns = b_ag[2] - b_ns[2]
            actual_late, actual_ag = res["Late fusion (fixed w)"][0], res["AGREE rerank (ours)"][0]
            actual_tw, actual_ns = res["AGREE rerank, test-set-wide multiset"][0], res["AGREE rerank, per half, 11 features (no event-structure context)"][0]
            per_seed["ci"] = {"aurc_diff": [p_d, lo_d, hi_d], "aurc_rel": [p_r, lo_r, hi_r], "aurc_diff_cc": [p_dc, lo_dc, hi_dc], "aurc_rel_cc": [p_rc, lo_rc, hi_rc], "aurc_diff_cal": [p_cal, lo_cal, hi_cal],
                              "map_diff": [actual_ag["mAP"] - actual_late["mAP"], float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                              "map_diff_testwide": [actual_tw["mAP"] - actual_late["mAP"], float(np.percentile(diffs_tw, 2.5)), float(np.percentile(diffs_tw, 97.5))],
                              "map_diff_ctx": [actual_ag["mAP"] - actual_ns["mAP"], float(np.percentile(diffs_ctx, 2.5)), float(np.percentile(diffs_ctx, 97.5))],
                              "unshown_diff_ctx": [actual_ag["unshown"] - actual_ns["unshown"], float(np.percentile(diffs_uns, 2.5)), float(np.percentile(diffs_uns, 97.5))], "n_boot_map": nb_map, "n_boot": nB}
            per_seed["omega"] = omega; per_seed["tau"] = tau; per_seed["lambda"] = lam; per_seed["tau0"] = tau0; per_seed["f1_valid_at_tau0"] = f1v; per_seed["n_cand_test"] = len(rows_t); per_seed["n_cand_valid"] = len(rows_v)
    # ---- leave-one-league-out (if models exist)
    league_rows = []
    for L in (LEAGUES if (args.export_seed in seeds and not args.no_league) else []):
        tv, ta_ = f"visual_s0_xl-{L}", f"audio_s0_xl-{L}"
        if not (os.path.exists(os.path.join(RESULTS, "preds", tv)) and os.path.exists(os.path.join(RESULTS, "preds", ta_))): continue
        tg = [g for g in test_games if league_of(g) == L and os.path.exists(os.path.join(RESULTS, "preds", tv, g, "2.npy"))]
        vg = [g for g in val_games if league_of(g) != L and os.path.exists(os.path.join(RESULTS, "preds", tv, g, "2.npy"))]
        if len(tg) < 3 or len(vg) < 5: continue
        GL, GLv = GameSet(tg), GameSet(vg)
        Zv, Za = [load_logits(tv, g, h) for g in GL.games for h in (1, 2)], [load_logits(ta_, g, h) for g in GL.games for h in (1, 2)]
        Zvv, Zav = [load_logits(tv, g, h) for g in GLv.games for h in (1, 2)], [load_logits(ta_, g, h) for g in GLv.games for h in (1, 2)]
        Pv, Pa, Pvv, Pav = [sigmoid(z) for z in Zv], [sigmoid(z) for z in Za], [sigmoid(z) for z in Zvv], [sigmoid(z) for z in Zav]
        om = tune_omega(GLv, Pvv, Pav); Pf, Pfv = late_fusion(Pv, Pa, om), late_fusion(Pvv, Pav, om)
        t0L, _ = f1_optimal_threshold(GLv, Pfv); rv, rt = build_candidates(GLv, Pfv, Pvv, Pav, thr=t0L), build_candidates(GL, Pf, Pv, Pa, thr=t0L)
        cb = combiner_by_seed.get(args.export_seed, {"class_inter": False, "C": 1.0}); rb = rerank_by_seed.get(args.export_seed, {"thr": 0.1, "class_inter": False, "C": 1.0})
        sf = fit_reliability(rv, FULL_R, False, class_inter=cb["class_inter"], C=cb["C"]); sc_ = fit_reliability(rv, ["p_f"], False)
        rv_r, rt_r = build_candidates(GLv, Pfv, Pvv, Pav, thr=rb["thr"]), build_candidates(GL, Pf, Pv, Pa, thr=rb["thr"]); sr = fit_reliability(rv_r, FULL_R, True, class_inter=rb["class_inter"], C=rb["C"], cross_fit=False)
        r_ = sf(rt); y_ = np.array([r["correct"] for r in rt])
        league_rows.append({"league": L, "n_test_games": len(tg), "loose_late": GL.mAP([nms_detections(P) for P in Pf], "loose")["mAP"],
                            "loose_agree": GL.mAP(rerank_score_multiset_per_half(GL, Pf, rt_r, sr.raw(rt_r)), "loose")["mAP"],
                            "AURC_conf": risk_coverage(sc_(rt), y_)[2], "AURC_agree": risk_coverage(r_, y_)[2], "n_cand": len(rt)})
    # ---- write outputs
    write_csv(os.path.join(OUT, f"main_results{tag}.csv"), main_rows); write_csv(os.path.join(OUT, f"selective{tag}.csv"), sel_rows)
    write_csv(os.path.join(OUT, f"diagnostic{tag}.csv"), diag_rows); write_csv(os.path.join(OUT, f"ablation{tag}.csv"), abl_rows, fields=["seed", "ablation", "setting", "AURC", "risk80", "loose", "AURC_conf"])
    write_csv(os.path.join(OUT, f"rare{tag}.csv"), rare_rows); write_csv(os.path.join(OUT, f"degradation{tag}.csv"), deg_rows)
    if league_rows or not tag: write_csv(os.path.join(OUT, f"league{tag}.csv"), league_rows)
    summary = {"seeds": seeds, "n_valid_games": len(val_games), "n_test_games": len(test_games), "rare_classes": [CLASSES[c] for c in rare],
               "class_counts_train": {CLASSES[c]: int(counts[c]) for c in range(NUM_CLASSES)}, "tau0_by_seed": tau0_by_seed, "omega_by_seed": omega_by_seed, "tau_by_seed": tau_by_seed, "lambda_by_seed": lambda_by_seed,
               "combiner_by_seed": combiner_by_seed, "rerank_by_seed": rerank_by_seed, "rare_val_by_seed": rare_val_by_seed,
               "validation_folds": "five-fold GroupKFold by match", "aurc_ties": "expected risk under uniform ordering within equal-score blocks",
               "analysis_started_utc": started_utc, "analysis_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "analysis_command": " ".join(sys.argv), **per_seed}
    json.dump(summary, open(os.path.join(OUT, f"summary{tag}.json"), "w"), indent=2); log("done")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--n-boot-map", type=int, default=1000); p.add_argument("--fast", action="store_true")
    p.add_argument("--ens-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="seed pool for the same-type ensembles (must all have predictions)")
    p.add_argument("--out-tag", default="", help="suffix for output files, e.g. _s2 when running one seed per process")
    p.add_argument("--export-seed", type=int, default=0, help="seed whose candidates are exported as SN-Agree and used for the bootstrap CIs and league table")
    p.add_argument("--no-league", action="store_true"); p.add_argument("--out-dir", default="", help="subdirectory of results/ for the outputs (e.g. v3)")
    main(p.parse_args())
