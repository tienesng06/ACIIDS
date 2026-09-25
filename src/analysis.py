"""Core analysis for SN-Agree: fusion baselines, candidate generation, agreement scores, reliability
calibration, reranking, abstention, mAP (official SoccerNet code, in-memory), selective metrics, bootstrap.
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from SoccerNet.Evaluation.ActionSpotting import average_mAP
from sklearn.isotonic import IsotonicRegression
from scipy.ndimage import maximum_filter1d

NMS_WIN = 20          # frames (10 s at 2 fps) for per-class temporal NMS
CAND_THR = 0.10       # base confidence threshold defining the candidate set
CORR_DELTA = 10       # frames (5 s): a candidate is 'correct' if a same-class event lies within +-5 s
LOCAL_WIN = 2         # frames: branch score for a candidate = max within +-1 s
RARE_PRIOR = 0.01     # classes with training prior below this are 'rare'
LOOSE_WIN = 120       # frames (60 s): widest loose-mAP tolerance; used to define 'nearest event' visibility of a candidate
CTX_WINDOWS = (15, 30)  # seconds: wider branch context (mean before / mean after / max) attached to every candidate
PREV_WIN = 60         # seconds: 'which class fired in the preceding minute' (max fused score over [t-60 s, t-5 s])
PREV_GAP = 10         # frames (5 s) excluded before t in the preceding-minute window

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z.astype(np.float32)))
def softmax_events(z):
    """QMF fused logits over 17 events + background -> per-event probabilities (background column dropped)."""
    z = z.astype(np.float32); z = z - z.max(-1, keepdims=True); p = np.exp(z); p /= p.sum(-1, keepdims=True); return p[:, :NUM_CLASSES]
def logsumexp(z, axis=-1):
    m = z.max(axis=axis, keepdims=True); return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)

def load_logits(tag, game, half):
    return np.load(os.path.join(RESULTS, "preds", tag, game, f"{half}.npy")).astype(np.float32)

class GameSet:
    """Holds ground truth (official format arrays) for a list of games."""
    def __init__(self, games):
        self.games = games; self.targets, self.closests, self.events, self.nframes = [], [], {}, {}
        for g in games:
            n = {h: np.load(visual_path(g, h), mmap_mode="r").shape[0] for h in (1, 2)}
            ev = label_frames(g, n); self.events[g] = ev; self.nframes[g] = n
            for h in (1, 2):
                t = np.zeros((n[h], NUM_CLASSES), dtype=np.int64)
                for f, c, vis in ev[h]: t[f, c] = 1 if vis else -1   # official convention: -1 marks unshown
                self.targets.append(t); self.closests.append(self._closest(t))
    @staticmethod
    def _closest(t):
        c = np.zeros_like(t) - 1
        for k in range(t.shape[-1]):
            idx = np.where(t[:, k] != 0)[0].tolist()
            if len(idx) == 0: continue
            idx.insert(0, -idx[0]); idx.append(2 * t.shape[0])
            for i in np.arange(len(idx) - 2) + 1:
                s, e = max(0, (idx[i - 1] + idx[i]) // 2), min(t.shape[0], (idx[i] + idx[i + 1]) // 2)
                c[s:e, k] = t[idx[i], k]
        return c
    def mAP(self, det_list, metric="loose", subset=None):
        """det_list: list aligned with self.targets of (n, C) score arrays (post-NMS, zeros elsewhere)."""
        deltas = np.arange(12) * 5 + 5 if metric == "loose" else np.arange(5) * 1 + 1   # seconds; average_mAP multiplies by framerate
        T, C, D = self.targets, self.closests, det_list
        if subset is not None:
            T = [T[i] for i in subset]; C = [C[i] for i in subset]; D = [D[i] for i in subset]
        a, pc, av, pcv, au, pcu = average_mAP(T, D, C, FPS, deltas=deltas)
        return {"mAP": float(a), "per_class": [float(x) for x in pc], "visible": float(av), "unshown": float(au), "per_class_visible": [float(x) for x in pcv], "per_class_unshown": [float(x) for x in pcu]}

def per_frame_probs(tag_or_fn, gs, half_iter=None):
    """Return list of (n, C) probability arrays aligned with gs.targets."""
    out = []
    for g in gs.games:
        for h in (1, 2):
            out.append(sigmoid(load_logits(tag_or_fn, g, h)) if isinstance(tag_or_fn, str) else tag_or_fn(g, h))
    return out

def nms_detections(P):
    """Per-class NMS on (n, C) probabilities -> (n, C) sparse score array."""
    D = np.full_like(P, -1.0)   # official convention: -1 where there is no detection
    for c in range(P.shape[1]):
        for f, s in nms_1d(P[:, c], NMS_WIN, thresh=0.0): D[f, c] = s
    return D

# ---------- fusion baselines ----------
def late_fusion(Pv, Pa, w): return [w * a + (1 - w) * b for a, b in zip(Pv, Pa)]
def mean_ensemble(P_list): return [np.mean([P[i] for P in P_list], axis=0) for i in range(len(P_list[0]))]

def energy_weighted_fusion(Zv, Za, tau):
    """QMF-style dynamic late fusion: per-frame weights from energy-based uncertainty E = -logsumexp(z)."""
    out = []
    for zv, za in zip(Zv, Za):
        ev, ea = -logsumexp(zv, -1), -logsumexp(za, -1)
        wv = np.exp(-ev / tau); wa = np.exp(-ea / tau); s = wv + wa + 1e-12
        out.append((wv / s)[:, None] * sigmoid(zv) + (wa / s)[:, None] * sigmoid(za))
    return out

def agreement_ensemble(D_list, cluster_win=1, min_models=2, bypass_classes=()):
    """PAVE Weighted Event Fusion (Altawijri & Mathkour 2026, no code released; re-implemented from the published rule):
    per-model post-NMS detections are clustered within +-12 frames at 25 fps (= +-0.5 s = +-1 frame at 2 fps), clusters
    with fewer than min_models=2 members are discarded, the cluster score is the mean over contributing models times
    sqrt(n/N), and bypass classes (their tackle exception) keep single-model detections."""
    N = len(D_list); out = np.zeros_like(D_list[0])
    for c in range(out.shape[1]):
        items = []
        for m, D in enumerate(D_list):
            for f in np.where(D[:, c] > 0)[0]: items.append((int(f), float(D[f, c]), m))
        items.sort(key=lambda x: -x[1]); used = [False] * len(items)
        for i, (f, s, m) in enumerate(items):
            if used[i]: continue
            members = [(f, s, m)]; used[i] = True
            for j in range(i + 1, len(items)):
                if not used[j] and abs(items[j][0] - f) <= cluster_win and items[j][2] not in [x[2] for x in members]:
                    members.append(items[j]); used[j] = True
            n = len(members)
            if n >= min_models or c in bypass_classes:
                out[f, c] = np.mean([x[1] for x in members]) * math.sqrt(n / N)
    return out

# ---------- candidates and agreement ----------
def local_max(P, f, c):
    lo, hi = max(0, f - LOCAL_WIN), min(P.shape[0], f + LOCAL_WIN + 1); return float(P[lo:hi, c].max())

def js_agreement(pv, pa):
    """1 - JS divergence (base 2) between the two per-class Bernoulli vectors, averaged over classes."""
    eps = 1e-6; pv = np.clip(pv, eps, 1 - eps); pa = np.clip(pa, eps, 1 - eps); m = 0.5 * (pv + pa)
    def kl(p, q): return p * np.log2(p / q) + (1 - p) * np.log2((1 - p) / (1 - q))
    js = 0.5 * kl(pv, m) + 0.5 * kl(pa, m)
    return float(1.0 - js.mean())

_ctx_cache = {}
def context_arrays(Pv, Pa, Pf):
    """Event-structure context of one half (post-hoc, from the score arrays and the clock), computed once per half:
    for each branch and W in CTX_WINDOWS, the mean score over the W s before t, the mean over the W s after t and the
    max over +-W s; seconds since the half started (exp(-t/20 s) and min(t, 600 s)/600); and, for every class, the max
    fused score over the preceding minute [t-60 s, t-5 s]. Motivation: src/unshown_study.py -- off-camera restarts
    (kick-off after a goal or at the half start, free-kick after a foul) are scored from context by window-label
    detectors such as NetVLAD++ but not by frame-level branches; these features let the reranker do the same."""
    key = (id(Pv), id(Pa), id(Pf), Pf.shape[0], round(float(Pf.sum()), 3), round(float(Pv.sum()), 3), round(float(Pa.sum()), 3))
    if key in _ctx_cache: return _ctx_cache[key]
    n = Pf.shape[0]; t = np.arange(n); ex = {}
    for name, P in (("v", Pv), ("a", Pa)):
        cs = np.vstack([np.zeros((1, P.shape[1])), np.cumsum(P.astype(np.float64), axis=0)])
        for Ws in CTX_WINDOWS:
            W = Ws * FPS
            lo, hi = np.maximum(0, t - W), t; ex[f"p_{name}_pre{Ws}"] = ((cs[hi] - cs[lo]) / np.maximum(1, hi - lo)[:, None]).astype(np.float32)
            lo2, hi2 = np.minimum(n, t + 1), np.minimum(n, t + W + 1); ex[f"p_{name}_post{Ws}"] = ((cs[hi2] - cs[lo2]) / np.maximum(1, hi2 - lo2)[:, None]).astype(np.float32)
            ex[f"p_{name}_max{Ws}"] = maximum_filter1d(P, size=2 * W + 1, axis=0, mode="nearest").astype(np.float32)
            # Past-only maximum over [t-W, t]. Unlike p_*_max*, this is available when the candidate fires.
            ex[f"p_{name}_premax{Ws}"] = maximum_filter1d(P, size=W + 1, axis=0, mode="nearest", origin=W // 2).astype(np.float32)
    ts = t / FPS; ex["t_half_exp20"] = np.exp(-ts / 20.0).astype(np.float32); ex["t_half_cap600"] = (np.minimum(ts, 600.0) / 600.0).astype(np.float32)
    W = PREV_WIN * FPS; mx = maximum_filter1d(Pf, size=W, axis=0, mode="nearest", origin=(W // 2) - 1)   # window [t-W+1, t]
    sh = np.vstack([np.zeros((PREV_GAP, Pf.shape[1]), Pf.dtype), mx[:-PREV_GAP]])                     # shifted PREV_GAP frames back
    for c in range(Pf.shape[1]): ex[f"prev{PREV_WIN}_{c}"] = sh[:, c].astype(np.float32)
    if len(_ctx_cache) > 1200: _ctx_cache.clear()   # ~4.5 MB per half; bounded so the degradation and league loops cannot grow it without limit
    _ctx_cache[key] = ex; return ex

def match_candidates_one_to_one(D, events, thr, delta=CORR_DELTA):
    """Match thresholded detections to same-class events one-to-one within +-delta.

    This mirrors SoccerNet's per-half/per-class convention: visit ground-truth events in temporal order and assign
    the highest-scoring still-unmatched detection in the tolerance window. The returned mapping is
    (frame, class) -> event visibility. A slow reference implementation in tests/test_analysis.py checks ties,
    duplicate candidates, adjacent events and boundary cases.
    """
    matched = {}
    for c in range(NUM_CLASSES):
        det = [int(f) for f in np.where(D[:, c] >= thr)[0]]
        used = set()
        for gf, gc, vis in sorted((e for e in events if e[1] == c), key=lambda e: e[0]):
            eligible = [f for f in det if f not in used and abs(f - gf) <= delta]
            if not eligible:
                continue
            # np.argmax/official matching resolves equal scores by temporal index.
            f = max(eligible, key=lambda x: (float(D[x, c]), -x))
            used.add(f); matched[(f, c)] = vis
    return matched


def build_candidates(gs, Pf, Pv, Pa, thr=CAND_THR, extra=None):
    """Candidate events from fused detections with one-to-one +-5 s correctness labels and branch evidence."""
    rows = []; k = 0
    for g in gs.games:
        for h in (1, 2):
            D = nms_detections(Pf[k]); ev = gs.events[g][h]; cx = context_arrays(Pv[k], Pa[k], Pf[k])
            official_matches = match_candidates_one_to_one(D, ev, thr)
            gt = {}
            for f, c, vis in ev: gt.setdefault(c, []).append((f, vis))
            for c in range(NUM_CLASSES):
                for f in np.where(D[:, c] >= thr)[0]:  # D is -1 where no detection
                    pf = float(D[f, c]); pv = local_max(Pv[k], f, c); pa = local_max(Pa[k], f, c)
                    # temporal context of each branch over +-CORR_DELTA (5 s): peak offset between branches, max and mean score
                    lo5, hi5 = max(0, f - CORR_DELTA), min(Pv[k].shape[0], f + CORR_DELTA + 1); wv, wa = Pv[k][lo5:hi5, c], Pa[k][lo5:hi5, c]
                    ctx = {"dt_peak": abs(int(np.argmax(wv)) - int(np.argmax(wa))) / FPS, "p_v_5s": float(wv.max()), "p_a_5s": float(wa.max()), "p_v_mean5s": float(wv.mean()), "p_a_mean5s": float(wa.mean())}
                    match = [(abs(ff - f), vis) for ff, vis in gt.get(c, [])]
                    correct, vis, vis_near = int((f, c) in official_matches), official_matches.get((f, c), -1), -1
                    if match:
                        # 'visible': visibility of the matched event (only defined when correct); 'vis_near': visibility of the
                        # nearest same-class event within the widest loose tolerance (60 s), defined for correct AND wrong
                        # candidates, so that stratifying by it is not degenerate.
                        dmin, v = min(match); vis_near = v if dmin <= LOOSE_WIN else -1
                    lo1 = max(0, f - LOCAL_WIN); lo5p = max(0, f - CORR_DELTA)
                    wv1, wa1 = Pv[k][lo1:f + 1, c], Pa[k][lo1:f + 1, c]
                    wv5p, wa5p = Pv[k][lo5p:f + 1, c], Pa[k][lo5p:f + 1, c]
                    row = {"game": g, "half": h, "frame": int(f), "cls": c, "p_f": pf, "p_v": pv, "p_a": pa,
                           "A_top": float(int(np.argmax(Pv[k][f]) == np.argmax(Pa[k][f]))),
                           "A_gap": 1.0 - abs(pv - pa), "A_js": js_agreement(Pv[k][f], Pa[k][f]), "A_prod": math.sqrt(max(pv, 0) * max(pa, 0)),
                           "p_v_past1": float(wv1.max()), "p_a_past1": float(wa1.max()),
                           "p_v_past5": float(wv5p.max()), "p_a_past5": float(wa5p.max()),
                           "p_v_meanpast5": float(wv5p.mean()), "p_a_meanpast5": float(wa5p.mean()),
                           "dt_peak_past": abs(int(np.argmax(wv5p)) - int(np.argmax(wa5p))) / FPS,
                           "correct": correct, "visible": vis, "vis_near": vis_near, **ctx}
                    for name, arr in cx.items(): row[name] = float(arr[f, c]) if arr.ndim == 2 else float(arr[f])
                    if extra:
                        for name, arr in extra.items(): row[name] = float(arr[k][f, c]) if arr[k].ndim == 2 else float(arr[k][f])
                    rows.append(row)
            k += 1
    return rows

def f1_optimal_threshold(gs, Pf, grid=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    """Threshold on fused detections maximising F1 at +-CORR_DELTA on the validation split (the system's operating point)."""
    best = (-1.0, grid[0]); k = 0; per_half = []
    for g in gs.games:
        for h in (1, 2):
            D = nms_detections(Pf[k]); ev = gs.events[g][h]
            per_half.append((D, ev, len(ev))); k += 1
    for thr in grid:
        tp = fp = 0; n_gt = 0
        for D, ev, ng in per_half:
            n_gt += ng
            matches = match_candidates_one_to_one(D, ev, thr)
            nd = sum(int((D[:, c] >= thr).sum()) for c in range(NUM_CLASSES))
            tp += len(matches); fp += nd - len(matches)
        prec = tp / max(1, tp + fp); rec = min(1.0, tp / max(1, n_gt)); f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if f1 > best[0]: best = (f1, thr)
    return best[1], best[0]

# ---------- error correlation on an unselected set ----------
def unselected_error_set(gs, seed=0):
    """A candidate-independent evaluation set for branch error correlation: every ground-truth event of every half
    (positives) plus an equal number of random (frame, class) negatives per half, with the class drawn from that half's
    event classes and the frame at least CORR_DELTA away from any same-class event. Returns list of (k, frame, cls, y)."""
    rng = np.random.RandomState(seed); items = []; k = 0
    for g in gs.games:
        for h in (1, 2):
            ev = gs.events[g][h]; n = gs.nframes[g][h]; by_c = {}
            for f, c, _ in ev: by_c.setdefault(c, []).append(f); items.append((k, int(f), int(c), 1))
            for _, c, _ in ev:
                for _try in range(50):
                    f = int(rng.randint(0, n))
                    if all(abs(f - ff) > CORR_DELTA for ff in by_c.get(c, [])): items.append((k, f, c, 0)); break
            k += 1
    return items

def branch_errors(P_list, items, thr=0.5):
    """Error indicator per item for one branch: a positive is an error if no same-class score >= thr lies within
    +-CORR_DELTA; a negative is an error if one does."""
    err = np.zeros(len(items), dtype=float)
    for i, (k, f, c, y) in enumerate(items):
        lo, hi = max(0, f - CORR_DELTA), min(P_list[k].shape[0], f + CORR_DELTA + 1)
        fired = bool(P_list[k][lo:hi, c].max() >= thr); err[i] = float(fired != bool(y))
    return err

def phi(e1, e2):
    if np.std(e1) == 0 or np.std(e2) == 0: return float("nan")
    return float(np.corrcoef(e1, e2)[0, 1])

# ---------- reliability calibration ----------
class Reliability:
    """Isotonic P(correct | score) with class-conditional curves (n_c >= n_min) shrunk toward the pooled curve."""
    def __init__(self, n_min=30, shrink=20.0): self.n_min, self.shrink = n_min, shrink; self.pooled = None; self.per_class = {}
    def fit(self, x, y, cls):
        x, y, cls = np.asarray(x, float), np.asarray(y, float), np.asarray(cls)
        self.pooled = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(x, y)
        for c in np.unique(cls):
            m = cls == c
            if m.sum() >= self.n_min:
                self.per_class[int(c)] = (IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(x[m], y[m]), int(m.sum()))
        return self
    def predict(self, x, cls, class_conditional=True):
        x, cls = np.asarray(x, float), np.asarray(cls); r = self.pooled.predict(x)
        if class_conditional:
            for c, (iso, n) in self.per_class.items():
                m = cls == c
                if m.any():
                    lam = n / (n + self.shrink); r[m] = lam * iso.predict(x[m]) + (1 - lam) * r[m]
        return np.clip(r, 0, 1)

def learned_agreement_features(rows):
    return np.stack([[r["p_v"], r["p_a"], r["A_gap"], r["A_js"], r["A_prod"], r["p_f"]] for r in rows])

# ---------- selective prediction metrics ----------
def risk_coverage(score, correct):
    """Tie-aware risk--coverage curve and AURC.

    Within each equal-score block, risk at an intermediate prefix is its expectation under a uniformly random
    ordering of that block. This removes dependence on candidate input order after isotonic calibration.
    """
    score, correct = np.asarray(score, float), np.asarray(correct, float)
    if len(score) == 0:
        return np.array([]), np.array([]), float("nan")
    o = np.argsort(-score, kind="stable"); s = score[o]; err = 1 - correct[o]
    risk = np.zeros(len(err), dtype=float); prev_err = 0.0; i = 0
    while i < len(err):
        j = i + 1
        while j < len(err) and s[j] == s[i]: j += 1
        block_err = float(err[i:j].sum()); m = j - i
        u = np.arange(1, m + 1, dtype=float)
        risk[i:j] = (prev_err + u * block_err / m) / (i + u)
        prev_err += block_err; i = j
    cov = (np.arange(len(err)) + 1) / len(err)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return cov, risk, float(trap(risk, cov))


def risk_coverage_tie_bounds(score, correct):
    """Best/worst empirical AURC attainable by ordering errors last/first inside equal-score blocks."""
    score, correct = np.asarray(score, float), np.asarray(correct, float)
    if len(score) == 0: return float("nan"), float("nan")
    o = np.argsort(-score, kind="stable"); s = score[o]; err = 1 - correct[o]
    best, worst = np.zeros(len(err)), np.zeros(len(err)); eb = ew = 0.0; i = 0
    while i < len(err):
        j = i + 1
        while j < len(err) and s[j] == s[i]: j += 1
        q, m = int(err[i:j].sum()), j - i
        bseq = np.r_[np.zeros(m - q), np.ones(q)]; wseq = np.r_[np.ones(q), np.zeros(m - q)]
        best[i:j] = (eb + np.cumsum(bseq)) / np.arange(i + 1, j + 1)
        worst[i:j] = (ew + np.cumsum(wseq)) / np.arange(i + 1, j + 1)
        eb += q; ew += q; i = j
    cov = (np.arange(len(err)) + 1) / len(err); trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trap(best, cov)), float(trap(worst, cov))

def risk_at_coverage(score, correct, cov_target):
    cov, risk, _ = risk_coverage(score, correct); i = np.searchsorted(cov, cov_target); return float(risk[min(i, len(risk) - 1)])


def acceptance_probability(score, cov_target):
    """Expected inclusion at a target coverage under uniform random tie breaking."""
    score = np.asarray(score, float); n = len(score)
    if n == 0: return np.array([], float)
    target = float(np.clip(cov_target, 0, 1)) * n
    out = np.zeros(n, float)
    o = np.argsort(-score, kind="stable"); s = score[o]; used = 0; i = 0
    while i < n and used < target:
        j = i + 1
        while j < n and s[j] == s[i]: j += 1
        take = min(j - i, target - used)
        out[o[i:j]] = take / (j - i)
        used += take; i = j
    return out

def ece(prob, correct, bins=10):
    prob, correct = np.asarray(prob, float), np.asarray(correct, float); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (prob > edges[i]) & (prob <= edges[i + 1]) if i > 0 else (prob >= edges[i]) & (prob <= edges[i + 1])
        if m.any(): e += m.mean() * abs(prob[m].mean() - correct[m].mean())
    return float(e)


def brier(prob, correct):
    """Mean squared probability error; lower is better."""
    return float(np.mean((np.asarray(prob, float) - np.asarray(correct, float)) ** 2))

def spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0: return float("nan"), float("nan")
    r, p = spearmanr(a, b); return float(r), float(p)

def bootstrap_by_game(rows, fn, n_boot=1000, seed=0):
    """Paired bootstrap over games: fn(list_of_rows) -> scalar. Returns (point, lo, hi)."""
    rng = np.random.RandomState(seed); games = sorted(set(r["game"] for r in rows)); by = {g: [] for g in games}
    for r in rows: by[r["game"]].append(r)
    point = fn(rows); vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(games), size=len(games), replace=True)
        sub = [x for i in pick for x in by[games[i]]]; vals.append(fn(sub))
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def rerank_score_multiset_per_half(gs, Pf, rows, score):
    """Score-multiset-preserving per-half variant: p_f values are reassigned within each (game, half, class), so a half can be processed
    on its own; the test-set-wide variant below reassigns within each class over the whole evaluated set (transductive)."""
    out = [nms_detections(P) for P in Pf]; idx = {}; k = 0
    for g in gs.games:
        for h in (1, 2): idx[(g, h)] = k; k += 1
    key = np.array([idx[(r["game"], r["half"])] * 100 + r["cls"] for r in rows]); pf = np.array([r["p_f"] for r in rows]); score = np.asarray(score, float)
    for kk in np.unique(key):
        m = np.where(key == kk)[0]; vals = np.sort(pf[m])[::-1]; order = m[np.argsort(-score[m], kind="stable")]
        for i, v in zip(order, vals): out[idx[(rows[i]["game"], rows[i]["half"])]][rows[i]["frame"], rows[i]["cls"]] = v
    return out

def rerank_score_multiset(gs, Pf, rows, score):
    """Score-multiset-preserving reranking: candidates keep the multiset of their fused scores p_f, reassigned in
    the order of `score` (highest score gets the highest p_f). Non-candidate detections are untouched. The official
    average-mAP samples the precision-recall curve at 200 fixed score thresholds, so it depends on the score
    distribution and not only on the order; keeping the distribution makes the comparison with the fused baseline a
    comparison of orderings only (see src/rerank_study*.py for the diagnosis of the score-compressing variant)."""
    out = [nms_detections(P) for P in Pf]; idx = {}; k = 0
    for g in gs.games:
        for h in (1, 2): idx[(g, h)] = k; k += 1
    cls = np.array([r["cls"] for r in rows]); pf = np.array([r["p_f"] for r in rows]); score = np.asarray(score, float)
    for c in np.unique(cls):
        m = np.where(cls == c)[0]; vals = np.sort(pf[m])[::-1]; order = m[np.argsort(-score[m], kind="stable")]
        for i, v in zip(order, vals): out[idx[(rows[i]["game"], rows[i]["half"])]][rows[i]["frame"], c] = v
    return out

# ---------- fast, semantics-preserving replacement for the official per-class matching ----------
def fast_compute_class_scores(target, closest, detection, delta):
    """Same output as SoccerNet.Evaluation.ActionSpotting.compute_class_scores, with the GT x prediction
    Python double loop replaced by searchsorted windows. Greedy in GT index order, highest unmatched score
    within +-delta/2, exactly as the official code."""
    gt_indexes = np.where(target != 0)[0]
    n_vis = int((target > 0).sum()); n_uns = int((target < 0).sum())
    pred_indexes = np.where(detection >= 0)[0]
    pred_scores = detection[pred_indexes]
    game_detections = np.zeros((len(pred_indexes), 3))
    game_detections[:, 0] = pred_scores
    game_detections[:, 2] = closest[pred_indexes]
    if len(pred_indexes) == 0 or len(gt_indexes) == 0:
        return game_detections, n_vis, n_uns
    matched = np.zeros(len(pred_indexes), dtype=bool); half = delta / 2
    lo_all = np.searchsorted(pred_indexes, gt_indexes - half, side="left")
    hi_all = np.searchsorted(pred_indexes, gt_indexes + half, side="right")
    for gi, lo, hi in zip(gt_indexes, lo_all, hi_all):
        if hi <= lo: continue
        sc = pred_scores[lo:hi].copy(); sc[matched[lo:hi]] = -np.inf
        # official: strict '>' against max_score starting at -1 -> first max in index order, score must be > -1
        j = int(np.argmax(sc))
        if sc[j] > -1:
            matched[lo + j] = True; game_detections[lo + j, 1] = 1
    return game_detections, n_vis, n_uns

import SoccerNet.Evaluation.ActionSpotting as _AS
_AS.compute_class_scores = fast_compute_class_scores

def fast_compute_precision_recall_curve(targets, closests, detections, delta):
    """Vectorised equivalent of the official compute_precision_recall_curve (200 thresholds in [0,1];
    visible/unshown splits by the sign of the closest ground-truth flag; nan -> 0), same outputs."""
    num_classes = targets[0].shape[-1]; thresholds = np.linspace(0, 1, 200)
    outs = [np.zeros((200, num_classes)) for _ in range(6)]
    for c in range(num_classes):
        parts = [np.array([[-1.0, 0.0, 0.0]])]; n_vis = n_uns = 0
        for target, closest, detection in zip(targets, closests, detections):
            d, nv, nu = fast_compute_class_scores(target[:, c], closest[:, c], detection[:, c], delta)
            parts.append(d); n_vis += nv; n_uns += nu
        td = np.concatenate(parts, axis=0)
        def curve(scores, tp, n_gt):
            o = np.argsort(-scores, kind="stable"); s = scores[o]; t = np.cumsum(tp[o])
            # number of predictions with score >= thr, and their TP sum
            k = np.searchsorted(-s, -thresholds, side="right")          # count of s >= thr
            TP = np.where(k > 0, t[np.maximum(k - 1, 0)], 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                p = np.nan_to_num(TP / k); r = np.nan_to_num(TP / n_gt)
            return p, r
        sc_all = td[:, 0]; tp = td[:, 1]; cl = td[:, 2]
        sc_vis = np.where(cl <= 0.5, -1.0, sc_all); sc_uns = np.where(cl >= -0.5, -1.0, sc_all)
        for j, (sc, ng) in enumerate([(sc_all, n_vis + n_uns), (sc_vis, n_vis), (sc_uns, n_uns)]):
            p, r = curve(sc, tp, ng); outs[2 * j][:, c] = p; outs[2 * j + 1][:, c] = r
    for j in range(3):
        P, R = outs[2 * j], outs[2 * j + 1]
        for i in range(num_classes):
            idx = np.argsort(R[:, i]); P[:, i] = P[idx, i]; R[:, i] = R[idx, i]
    return outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]

_AS.compute_precision_recall_curve = fast_compute_precision_recall_curve


def bootstrap_maps_fast(gs, det_list, picks, metric="loose", batch_size=16):
    """Official SoccerNet average-mAP for many match-bootstrap samples.

    Candidate matching is precomputed per match. Bootstrap samples then combine
    per-match threshold counts. This is equivalent to repeating ``GameSet.mAP``
    on each resample, but avoids rematching every detection thousands of times.
    Returns arrays for overall, visible, plus unshown average-mAP.
    """
    deltas = (np.arange(12) * 5 + 5 if metric == "loose" else np.arange(5) + 1) * FPS
    thresholds = np.linspace(0, 1, 200); G, D, T, C = len(gs.games), len(deltas), 200, NUM_CLASSES
    counts = np.zeros((G, D, 3, T, C), np.float32)
    true_pos = np.zeros_like(counts); n_gt = np.zeros((G, 3, C), np.float32)
    for gi in range(G):
        halves = (2 * gi, 2 * gi + 1)
        for di, delta in enumerate(deltas):
            for c in range(C):
                parts = []; nv = nu = 0
                for hi in halves:
                    td, v, u = fast_compute_class_scores(gs.targets[hi][:, c], gs.closests[hi][:, c], det_list[hi][:, c], delta)
                    parts.append(td); nv += v; nu += u
                td = np.concatenate(parts, axis=0) if parts else np.empty((0, 3))
                if di == 0:
                    n_gt[gi, :, c] = (nv + nu, nv, nu)
                score, tp, close = td[:, 0], td[:, 1], td[:, 2]
                for si, sc in enumerate((score, np.where(close <= .5, -1., score), np.where(close >= -.5, -1., score))):
                    if len(sc) == 0: continue
                    order = np.argsort(-sc, kind="stable"); ss = sc[order]; cs = np.cumsum(tp[order])
                    k = np.searchsorted(-ss, -thresholds, side="right")
                    counts[gi, di, si, :, c] = k
                    true_pos[gi, di, si, :, c] = np.where(k > 0, cs[np.maximum(k - 1, 0)], 0)
    picks = np.asarray(picks, int); W = np.stack([np.bincount(p, minlength=G) for p in picks]).astype(np.float32)
    flat_k, flat_tp = counts.reshape(G, -1), true_pos.reshape(G, -1); flat_n = n_gt.reshape(G, -1)
    out = np.empty((len(picks), 3), float)
    for start in range(0, len(picks), batch_size):
        w = W[start:start + batch_size]; b = len(w)
        k = (w @ flat_k).reshape(b, D, 3, T, C).astype(float)
        tp = (w @ flat_tp).reshape(b, D, 3, T, C).astype(float)
        ng = (w @ flat_n).reshape(b, 3, C).astype(float)
        precision = np.divide(tp, k, out=np.zeros_like(tp), where=k > 0)
        recall = np.divide(tp, ng[:, None, :, None, :], out=np.zeros_like(tp), where=ng[:, None, :, None, :] > 0)
        ap = np.zeros((b, D, 3, C), float)
        for level in np.arange(11) / 10:
            ap += np.where(recall >= level, precision, 0).max(axis=3)
        curves = (ap / 11).mean(axis=3)
        out[start:start + b] = ((curves[:, :-1] + curves[:, 1:]) / 2).mean(axis=1)
    out[:, 2] *= 17 / 13
    return out[:, 0], out[:, 1], out[:, 2]
