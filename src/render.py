"""Render artifact CSVs into manuscript/generated/{tab_*.tex, fig_*.tex, macros.tex}.
Usage: python src/render.py            (from real results)
       python src/render.py --stub     (placeholder values, clearly marked, so the manuscript compiles before a run)
Every number in the manuscript comes from here; nothing is typed by hand.
"""
import os, sys, csv, json, argparse, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, RESULTS, CLASSES, NUM_CLASSES, LEAGUES
from analysis import risk_coverage
GEN = os.path.join(os.path.dirname(ROOT), "manuscript", "generated")
NAN = float("nan")

def read(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p): return []
    with open(p) as f: return [dict(r) for r in csv.DictReader(f)]
def fl(x):
    try: return float(x)
    except Exception: return NAN
def isnan(x): return x is None or (isinstance(x, float) and math.isnan(x))
def agg(rows, key, val, group):
    """group -> (mean, sd, n) of column val (nan-aware)."""
    out = {}
    for r in rows: out.setdefault(r[group], []).append(fl(r.get(val)))
    res = {}
    for k, v in out.items():
        v = [x for x in v if not math.isnan(x)]
        res[k] = (float(np.mean(v)), float(np.std(v)), len(v)) if v else (NAN, NAN, 0)
    return res
SD = "\\sd"   # \sd{x} typesets the standard deviation as a small grey subscript (defined in main.tex)
def pm(m, s, d=1, pct=True):
    """mean with the SD as a grey subscript, so the column reads as numbers and the spread stays legible but secondary."""
    if isnan(m): return "--"
    if pct: return f"{100*m:.{d}f}{SD}{{{100*s:.{d}f}}}"
    if m < 0: return f"$-{-m:.{d}f}${SD}{{{s:.{d}f}}}"
    return f"{m:.{d}f}{SD}{{{s:.{d}f}}}"
def f1(x, d=1, pct=True):
    if isnan(x): return "--"
    v = 100 * x if pct else x
    return f"$-{-v:.{d}f}$" if v < 0 else f"{v:.{d}f}"
def esc(s): return s.replace("_", "\\_").replace("%", "\\%").replace(">", "$>$")
def words(n):
    w = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    return w[n] if 0 <= n <= 10 else str(n)
def bold(s): return f"\\textbf{{{s}}}"
def mode(vals):
    vals = [v for v in vals if v is not None]
    return max(sorted(set(vals)), key=vals.count) if vals else None
def write(name, s):
    os.makedirs(GEN, exist_ok=True)
    with open(os.path.join(GEN, name), "w") as f: f.write(s)
def bold_best(cells, vals, better="min"):
    """Return cells with the best (by vals) bolded; values that display identically (same cell text) tie and are all bolded."""
    ok = [(i, v) for i, v in enumerate(vals) if not isnan(v)]
    if not ok: return cells
    best_i = (min if better == "min" else max)(ok, key=lambda t: t[1])[0]
    shown = lambda c: c.split(SD)[0].split("$\\pm$")[0]
    return [bold(c) if (not isnan(v) and shown(c) == shown(cells[best_i])) else c for c, v in zip(cells, vals)]

SYS = [("Visual-only", "Visual-only"), ("Audio-only", "Audio-only"), ("Late fusion (fixed w)", "Late fusion (fixed $\\omega$)"),
       ("Early fusion (joint)", "Early fusion (joint)"), ("Mean ensemble (4 visual seeds)", "Mean ensemble (4 visual seeds)"),
       ("Agreement ensemble (PAVE-style)", "PAVE ensembling~\\cite{altawijri2026pave}"),
       ("Dynamic fusion (QMF-style)", "Dynamic fusion (QMF-style)"), ("QMF (Zhang et al.)", "QMF~\\cite{zhang2023qmf}"), ("AGREE rerank (ours)", "\\method (ours)"), ("AGREE rerank, test-set-wide multiset", "\\method, transductive variant")]
SEL = [("Confidence-only (calibrated p_f)", "Confidence-only (calibrated $p_f$)"), ("Confidence-only, class-conditional", "Confidence-only, class-cond."),
       ("Same-type ensemble agreement", "PAVE cluster score~\\cite{altawijri2026pave}"), ("Dynamic fusion confidence (QMF-style)", "QMF-style fused confidence"),
       ("QMF fused confidence", "QMF confidence + isotonic~\\cite{zhang2023qmf}"), ("NetVLAD++ confidence", "NetVLAD++ confidence"),
       ("CALF confidence", "CALF confidence~\\cite{cioppa2020calf}"), ("AGREE, pooled calibration", "\\method (ours)"),
       ("AGREE, transductive score reassignment", "\\method, transductive variant")]
LEAGUE_NAMES = {"england_epl": "England (EPL)", "europe_uefa-champions-league": "UEFA Champions League", "france_ligue-1": "France (Ligue 1)",
                "germany_bundesliga": "Germany (Bundesliga)", "italy_serie-a": "Italy (Serie A)", "spain_laliga": "Spain (La Liga)"}
LEAGUE_SHORT = {"england_epl": "EPL", "europe_uefa-champions-league": "UEFA CL", "france_ligue-1": "Ligue 1", "germany_bundesliga": "Bundesliga", "italy_serie-a": "Serie A", "spain_laliga": "La Liga"}

def render(stub=False):
    S = json.load(open(os.path.join(RESULTS, "summary.json"))) if os.path.exists(os.path.join(RESULTS, "summary.json")) else {}
    main_rows, sel, diag, abl, rare, deg, lg = read("main_results.csv"), read("selective.csv"), read("diagnostic.csv"), read("ablation.csv"), read("rare.csv"), read("degradation.csv"), read("league.csv")
    sn = read("sn_agree_test.csv"); snv = read("sn_agree_valid.csv")
    M = {}; tag = "\\textcolor{red}{[PLACEHOLDER]}" if stub else ""
    seeds = S.get("seeds", [0, 1, 2, 3, 4]); M["nSeeds"] = str(len(seeds)); M["nTestGames"] = str(S.get("n_test_games", "--"))
    ci = S.get("ci", {}); M["nBoot"] = f"{ci.get('n_boot', 1000):,}"; M["nBootMap"] = f"{ci.get('n_boot_map', 200):,}"
    # ---- hyper-parameters (modal value over seeds + a note on unanimity)
    def hp(key, name, fmt=lambda v: f"{v:g}"):
        d = S.get(f"{key}_by_seed", {}); vals = list(d.values())
        if not vals and key in S: vals = [S[key]]
        if not vals: M[name] = "--"; M[name + "Note"] = "--"; return None
        m = mode(vals); k = vals.count(m); M[name] = fmt(m)
        M[name + "Note"] = "in every seed" if k == len(vals) else ("in every seed but one" if k == len(vals) - 1 else f"in {k} of {len(vals)} seeds")
        return m
    tau0 = hp("tau0", "tauZero"); omega = hp("omega", "omegaVal"); lam = hp("lambda", "lambdaVal"); hp("tau", "tauQmf")
    cbs = S.get("combiner_by_seed", {}); rbs = S.get("rerank_by_seed", {})
    if cbs:
        cis = [v["class_inter"] for v in cbs.values()]; M["combinerNote"] = ("class-specific coefficients were selected in every seed" if all(cis) else ("shared coefficients were selected in every seed" if not any(cis) else f"class-specific coefficients were selected in {sum(cis)} of {len(cis)} seeds"))
    if rbs:
        th = [v["thr"] for v in rbs.values()]; m = mode(th); M["tauR"] = f"{m:g}"; k = th.count(m); M["tauRNote"] = "in every seed" if k == len(th) else ("in every seed but one" if k == len(th) - 1 else f"in {k} of {len(th)} seeds")
        cis = [v["class_inter"] for v in rbs.values()]; M["rerankCombinerNote"] = ("class-specific coefficients in every seed" if all(cis) else ("shared coefficients in every seed" if not any(cis) else f"class-specific coefficients in {sum(cis)} of {len(cis)} seeds"))
    M["fOneValid"] = f"{S['f1_valid_at_tau0']:.2f}" if "f1_valid_at_tau0" in S else "--"
    M["nCandTest"] = f"{S['n_cand_test']:,}" if "n_cand_test" in S else "--"; M["nCandValid"] = f"{S['n_cand_valid']:,}" if "n_cand_valid" in S else "--"
    if sn: M["candPrecision"] = f1(np.mean([fl(r["correct"]) for r in sn]))
    if sel:
        ns = sorted(set(int(fl(r["n"])) for r in sel if r["selector"].startswith("AGREE, pooled")))
        if ns: M["nCandRange"] = f"{ns[0]:,}--{ns[-1]:,}"
    # ---------------- main table
    off_p = os.path.join(RESULTS, "baselines_official.json"); off_rows = json.load(open(off_p))["rows"] if os.path.exists(off_p) else []
    if off_rows:
        _nr = sorted({r["n_runs"] for r in off_rows})
        _seeds = (f"{_nr[0]} seeds each" if len(_nr) == 1 and _nr[0] > 1 else
                  "; ".join(f"{r['name']}: {'one seed' if r['n_runs'] == 1 else str(r['n_runs']) + ' seeds'}" for r in off_rows))
        OFFNOTE = f"The last two baselines are prior systems retrained with their official code on our features ({_seeds}). " if len(off_rows) > 1 else f"the last baseline is a prior system retrained with its official code on the same features ({_seeds}). "
        M["offSeeds"] = _seeds
    else: OFFNOTE = ""
    if main_rows:
        A = {k: agg(main_rows, "system", k, "system") for k in ("loose", "tight", "visible", "unshown")}
        present = [(k, n) for k, n in SYS if k in A["loose"]]
        cols = ("loose", "tight", "visible", "unshown")
        cells = {k: [pm(*A[c][k][:2]) for c in cols] for k, _ in present}
        for j, c in enumerate(cols):
            vals = [A[c][k][0] for k, _ in present]; col = bold_best([cells[k][j] for k, _ in present], vals, "max")
            for (k, _), cc in zip(present, col): cells[k][j] = cc
        lines = ["\\begin{table}[t]\\centering\\caption{Action spotting on the SoccerNet-v2 test split (average-mAP, \\%; mean$\\pm$SD over \\nSeeds{} seeds). " + OFFNOTE + tag + "}\\label{tab:main}",
                 "\\begin{tabular}{lcccc}\\toprule System & Loose & Tight & Visible & Unshown \\\\ \\midrule"]
        # prior systems first (ours last, after a rule), so the table reads baselines -> ours
        OURS = {"AGREE rerank (ours)", "AGREE rerank, test-set-wide multiset"}
        for k, n in present:
            if k not in OURS: lines.append(n + " & " + " & ".join(cells[k]) + " \\\\")
        off = os.path.join(RESULTS, "baselines_official.json")
        if os.path.exists(off) and json.load(open(off))["rows"]:
            for r in json.load(open(off))["rows"]:
                cell = (lambda v: pm(v[0], v[1])) if r["n_runs"] > 1 else (lambda v: f1(v[0]))
                lines.append(f"{r['name']}~\\cite{{{r['cite']}}} & {cell(r['loose'])} & {cell(r['tight'])} & {cell(r['loose_visible'])} & {cell(r['loose_unshown'])} \\\\")
                _n = "".join(ch for ch in r["name"] if ch.isalpha())
                M["map" + _n] = f1(r["loose"][0]); M["tmap" + _n] = f1(r["tight"][0]); M["unshown" + _n] = f1(r["loose_unshown"][0]); M["visible" + _n] = f1(r["loose_visible"][0])
                M["unshown" + _n + "SD"] = f1(r["loose_unshown"][1])
        # ---- unshown diagnostic (src/unshown_study.py): why a window-label baseline led on unshown events
        unp = os.path.join(RESULTS, "unshown_s0.json")
        if os.path.exists(unp):
            U = json.load(open(unp)); nb = U["nearest"]
            M["unsWithinFive"] = f"{100*nb['late_unshown']['within_5s']:.0f}\\%"; M["unsWithinFiveNV"] = f"{100*nb['netvlad_unshown']['within_5s']:.0f}\\%"
            M["unsMedianDist"] = f"{nb['late_unshown']['median_dist_s']:.1f}"; M["unsMedianDistNV"] = f"{nb['netvlad_unshown']['median_dist_s']:.1f}"
            pc = sorted(U["per_class"], key=lambda r: -r["n_unshown"])
            top = [r for r in pc if r["netvlad"] > r["late"]][:3]
            M["unsTopClasses"] = ", ".join(r["cls"].lower() for r in top)
            tot = sum(r["n_unshown"] for r in pc); M["unsTopShare"] = f"{100*sum(r['n_unshown'] for r in top)/max(1,tot):.0f}\\%"
            ko = next((r for r in pc if r["cls"] == "Kick-off"), None)
            if ko: M["unsKickoffOurs"] = f1(ko["late"]); M["unsKickoffNV"] = f1(ko["netvlad"]); M["unsKickoffN"] = str(ko["n_unshown"])
            for k, mk in (("kickoff_start_n", "unsKickoffStart"), ("kickoff_goal_n", "unsKickoffGoal")):
                if k in U: M[mk] = str(U[k])
            for k, mk in (("kickoff_start_score_ours", "unsKickoffScoreOurs"), ("kickoff_start_score_nv", "unsKickoffScoreNV")):
                if k in U: M[mk] = f"{U[k]:.2f}"
        # published results of stronger, video-input systems are quoted in the text (Section 6) rather than as table rows,
        # since they are not comparable at the input level; macros carry their ranges.
        lines.append("\\midrule")
        for k, n in present:
            if k in OURS: lines.append(n + " & " + " & ".join(cells[k]) + " \\\\")
        rep = os.path.join(RESULTS, "reported_sota.json")
        if os.path.exists(rep):
            rr = [r for r in json.load(open(rep))["rows"] if "NetVLAD" not in r["name"]]
            if rr:
                lo = [float(r["loose"]) for r in rr]; ti = [float(r["tight"]) for r in rr if r["tight"] is not None]
                M["sotaLooseRange"] = f"{min(lo):.1f}--{max(lo):.1f}"
                if ti: M["sotaTightRange"] = f"{min(ti):.1f}--{max(ti):.1f}"
                M["sotaCites"] = ",".join(r["cite"] for r in rr)
        lines += ["\\bottomrule\\end{tabular}\\end{table}"]; write("tab_main.tex", "\n".join(lines))
        mac = {"Visual-only": "Visual", "Audio-only": "Audio", "Late fusion (fixed w)": "Late", "Early fusion (joint)": "Joint", "Mean ensemble (4 visual seeds)": "MeanEns",
               "Agreement ensemble (PAVE-style)": "Ens", "Dynamic fusion (QMF-style)": "Qmf", "QMF (Zhang et al.)": "Qmf", "AGREE rerank (ours)": "Agree", "AGREE rerank, per half, 11 features (no event-structure context)": "AgreeNoCtx", "AGREE rerank, test-set-wide multiset": "AgreeTW", "AGREE rerank, score-compressing (p_f r^0.5)": "AgreeCompress"}
        for k, m in mac.items():
            if k in A["loose"]: M["map" + m] = f1(A["loose"][k][0]); M["tmap" + m] = f1(A["tight"][k][0]); M["unshown" + m] = f1(A["unshown"][k][0]); M["visible" + m] = f1(A["visible"][k][0])
        q = A["loose"].get("QMF (Zhang et al.)", A["loose"].get("Dynamic fusion (QMF-style)")); l = A["loose"].get("Late fusion (fixed w)")
        if q and l:
            d = round(100 * q[0], 1) - round(100 * l[0], 1); M["qmfVsLate"] = "matches" if abs(d) < 0.5 else (f"is {d:.1f} points above" if d > 0 else f"is {-d:.1f} points below")
        if "Agreement ensemble + rare bypass" in A["loose"] and "Agreement ensemble (PAVE-style)" in A["loose"]:
            d = 100 * (A["loose"]["Agreement ensemble + rare bypass"][0] - A["loose"]["Agreement ensemble (PAVE-style)"][0]); M["bypassDelta"] = "less than 0.05" if abs(d) < 0.05 else f"${d:+.1f}$"
    # ---------------- selective table
    if sel:
        # columns present in this run only: older result files carry no Brier or tie bounds, and the table
        # should render from whatever the pipeline produced rather than fail on a missing key.
        have = set(sel[0].keys()) if sel else set()
        cols = tuple(c for c in ("AURC", "risk80", "ECE", "Brier") if c in have)
        extra = tuple(c for c in ("risk90", "spearman", "AURC_tie_low", "AURC_tie_high") if c in have)
        A = {k: agg(sel, "selector", k, "selector") for k in cols + extra}
        present = [(k, n) for k, n in SEL if k in A["AURC"]]
        fmt = {"AURC": lambda m, s: pm(m, s, 3, False), "risk80": lambda m, s: pm(m, s, 1), "risk90": lambda m, s: pm(m, s, 1), "ECE": lambda m, s: pm(m, s, 3, False), "Brier": lambda m, s: pm(m, s, 3, False), "spearman": lambda m, s: pm(m, s, 2, False)}
        cells = {k: [fmt[c](*A[c][k][:2]) for c in cols] for k, _ in present}
        for j, c in enumerate(cols):
            vals = [A[c][k][0] for k, _ in present]; col = bold_best([cells[k][j] for k, _ in present], vals, "max" if c == "spearman" else "min")
            for (k, _), cc in zip(present, col): cells[k][j] = cc
        lines = ["\\begin{table}[t]\\centering\\caption{Selective prediction on test candidates (mean$\\pm$SD over \\nSeeds{} seeds, \\nCandRange{} per seed). AURC is the expectation over orderings within tied-score blocks; lower is better and risk is in \\%. The final row pools test matches for score reassignment, with probability calibration fitted on validation only; it is a transductive comparison.}\\label{tab:selective}\\resizebox{\\linewidth}{!}{%",
                 "\\begin{tabular}{l" + "c"*len(cols) + "}\\toprule Selector & " + " & ".join({"AURC": "AURC", "risk80": "Risk@80\\%", "ECE": "ECE", "Brier": "Brier"}[c] for c in cols) + " \\\\ \\midrule"]
        # rule before our own selectors, so the table reads baselines -> ours like Table~\ref{tab:main}
        _seen_ours = False
        for k, n in present:
            if not _seen_ours and k.startswith("AGREE"): lines.append("\\midrule"); _seen_ours = True
            lines.append(n + " & " + " & ".join(cells[k]) + " \\\\")
        lines += ["\\bottomrule\\end{tabular}}\\end{table}"]; write("tab_selective.tex", "\n".join(lines))
        mac = {"Confidence-only (calibrated p_f)": "Conf", "Same-type ensemble agreement": "Ens", "Dynamic fusion confidence (QMF-style)": "Qmf", "QMF fused confidence": "Qmf", "Confidence-only, class-conditional": "PfClass", "AGREE, pooled calibration": "Agree", "AGREE, class-conditional (ours)": "AgreeCC"}
        for k, m in mac.items():
            if k in A["AURC"]:
                M["aurc" + m] = f1(A["AURC"][k][0], 3, False); M["riskEighty" + m] = f1(A["risk80"][k][0]) + "\\%"; M["riskNinety" + m] = f1(A["risk90"][k][0]) + "\\%"
                M["ece" + m] = f1(A["ECE"][k][0], 3, False); M["rhoSel" + m] = f1(A["spearman"][k][0], 2, False)
                if "Brier" in A: M["brier" + m] = f1(A["Brier"][k][0], 3, False)
                if "AURC_tie_low" in A and "AURC_tie_high" in A and k in A["AURC_tie_low"] and k in A["AURC_tie_high"]:
                    M["tieSpan" + m] = f"{A['AURC_tie_high'][k][0] - A['AURC_tie_low'][k][0]:.3f}"
        c, a, cc = "Confidence-only (calibrated p_f)", "AGREE, pooled calibration", "Confidence-only, class-conditional"
        if c in A["AURC"] and a in A["AURC"]:
            M["aurcRel"] = f"{100*(A['AURC'][c][0]-A['AURC'][a][0])/A['AURC'][c][0]:.1f}"
        if cc in A["AURC"] and a in A["AURC"]:
            M["aurcRelCC"] = f"{100*(A['AURC'][cc][0]-A['AURC'][a][0])/A['AURC'][cc][0]:.1f}"
            M["aurcRelCal"] = f"{100*(A['AURC'][c][0]-A['AURC'][cc][0])/A['AURC'][c][0]:.1f}" if c in A["AURC"] else "--"
    # ---------------- diagnostic table + reliability figure
    if diag:
        A = {k: agg(diag, "feature", k, "feature") for k in ("spearman_all", "spearman_visible", "spearman_unshown")}
        names = [("A_top", "$A_{\\mathrm{top}}$ (corroboration: same top-1 class)"), ("A_prod", "$A_{\\mathrm{prod}}$ (corroboration: $\\sqrt{p_v p_a}$)"),
                 ("A_js", "$A_{\\mathrm{JS}}$ (consistency)"), ("A_gap", "$A_{\\mathrm{gap}}$ (consistency)"), None,
                 ("p_v", "$p_v$ (visual branch)"), ("p_a", "$p_a$ (audio branch)"), ("p_f", "$p_f$ (fused)")]
        strat = all(f in A["spearman_visible"] and not isnan(A["spearman_unshown"][f][0]) for f in ("A_top", "A_prod"))
        head = "Score & All & Near visible & Near unshown \\\\" if strat else "Score & Spearman $\\rho$ with correctness \\\\"
        cap = ("Diagnostic (RQ1). Top: Spearman correlation between each per-candidate score and correctness on the test candidates, overall and by whether the nearest same-class event within 60\\,s is visible or unshown (mean$\\pm$SD over \\nSeeds{} seeds). Bottom: error correlation $\\phi$ between branch decisions (score $\\ge0.5$ within $\\pm5$\\,s) on an unselected set of all test events plus an equal number of random negatives. " if strat else
               "Diagnostic (RQ1): Spearman correlation between each per-candidate score and the correctness label on the test candidates (mean$\\pm$SD over \\nSeeds{} seeds). Corroboration scores ($A_{\\mathrm{top}}$, $A_{\\mathrm{prod}}$) are informative; consistency scores ($A_{\\mathrm{gap}}$, $A_{\\mathrm{JS}}$) are weak or uninformative, as Proposition~\\ref{prop:corr} predicts. ")
        lines = ["\\begin{table}[t]\\centering\\caption{" + cap + tag + "}\\label{tab:diagnostic}", ("\\begin{tabular}{lccc}" if strat else "\\begin{tabular}{lc}") + "\\toprule " + head + " \\midrule"]
        for it in names:
            if it is None: lines.append("\\midrule"); continue
            f, nm = it
            if f not in A["spearman_all"]: continue
            cells = [pm(*A["spearman_all"][f][:2], 2, False)] + ([pm(*A["spearman_visible"][f][:2], 2, False), pm(*A["spearman_unshown"][f][:2], 2, False)] if strat else [])
            lines.append(nm + " & " + " & ".join(cells) + " \\\\")
        phi_rows = [("errcorr_unsel_visual_audio", "$\\phi$(visual, audio)"), ("errcorr_unsel_visual_visual", "$\\phi$(visual, visual seed)")]
        if all(f in A["spearman_all"] for f, _ in phi_rows):
            lines.append("\\midrule")
            for f, nm in phi_rows: lines.append(nm + " & " + pm(*A["spearman_all"][f][:2], 2, False) + (" & -- & --" if strat else "") + " \\\\")
            M["phiAV"] = f1(A["spearman_all"]["errcorr_unsel_visual_audio"][0], 2, False); M["phiVV"] = f1(A["spearman_all"]["errcorr_unsel_visual_visual"][0], 2, False)
            M["phiAVpos"] = f1(A["spearman_visible"]["errcorr_unsel_visual_audio"][0], 2, False); M["phiVVpos"] = f1(A["spearman_visible"]["errcorr_unsel_visual_visual"][0], 2, False)
            M["phiAVneg"] = f1(A["spearman_unshown"]["errcorr_unsel_visual_audio"][0], 2, False); M["phiVVneg"] = f1(A["spearman_unshown"]["errcorr_unsel_visual_visual"][0], 2, False)
            if "n_unsel" in A["spearman_all"]: M["nUnsel"] = f"{int(round(A['spearman_all']['n_unsel'][0])):,}"
            for f, m in [("errrate_unsel_visual", "V"), ("errrate_unsel_audio", "A"), ("errrate_unsel_visual_seed", "VV")]:
                if f in A["spearman_all"]: M["errRate" + m] = f1(A["spearman_all"][f][0]); M["missRate" + m] = f1(A["spearman_visible"][f][0])
        lines += ["\\bottomrule\\end{tabular}\\end{table}"]; write("tab_diagnostic.tex", "\n".join(lines))
        for f, m in [("A_top", "Top"), ("A_prod", "Prod"), ("A_js", "Js"), ("A_gap", "Gap"), ("p_v", "Pv"), ("p_a", "Pa"), ("p_f", "Pf")]:
            if f in A["spearman_all"]:
                M["rho" + m] = f1(A["spearman_all"][f][0], 2, False)
                if strat: M["rho" + m + "Vis"] = f1(A["spearman_visible"][f][0], 2, False); M["rho" + m + "Uns"] = f1(A["spearman_unshown"][f][0], 2, False)
        if sn and strat:
            vn = [fl(r.get("vis_near", "nan")) for r in sn]; n = len(vn)
            M["fracNearVisible"] = f1(sum(1 for v in vn if v == 1) / n); M["fracNearUnshown"] = f1(sum(1 for v in vn if v == 0) / n); M["fracNearNone"] = f1(sum(1 for v in vn if v == -1) / n)
        bins = [f"bin{b}" for b in range(10)]
        if all(b in A["spearman_all"] for b in bins):
            mean = [100 * A["spearman_all"][b][0] for b in bins]; sd = [100 * A["spearman_all"][b][1] for b in bins]
            coords = " ".join(f"({b+1},{mean[b]:.1f})" for b in range(10))
            M["binLow"] = f"{mean[0]:.0f}"; M["binHigh"] = f"{mean[9]:.0f}"; M["binSdMin"] = f"{min(sd):.1f}"; M["binSdMax"] = f"{max(sd):.1f}"; M["relBins"] = mean
            dip = min(range(2, 8), key=lambda b: mean[b]); M["binDip"] = f"{mean[dip]:.0f}"; M["binDipDecile"] = str(dip + 1)
            M["relPanel"] = ("\\begin{tikzpicture}\\begin{axis}[ybar, bar width=5pt, width=0.52\\linewidth, height=2.5cm, xlabel={Decile of $A_{\\mathrm{prod}}$ (1 = lowest)}, ylabel={Precision (\\%)}, xtick={1,...,10}, ymin=0, ymax=100, enlarge x limits=0.06, tick label style={font=\\scriptsize}, label style={font=\\scriptsize}, nodes near coords, nodes near coords style={font=\\tiny}, every node near coord/.append style={/pgf/number format/fixed, /pgf/number format/precision=0}]\n"
                              f"\\addplot coordinates {{{coords}}};\n" + "\\end{axis}\\end{tikzpicture}")
            write("fig_reliability.tex", "% reliability bins are the left panel of fig_rc.tex\n")
    # ---------------- two-panel figure: risk-coverage curves + calibration diagram
    if sn:
        y = np.array([fl(r["correct"]) for r in sn]); series = []; ymax = 0
        styles = {"r_conf": "figGray, dashed, line width=1.25pt", "r_conf_cc": "figTeal, dash dot, line width=1.25pt",
                  "r_agree": "figVisual, line width=1.85pt", "r_transductive": "figPurple, densely dotted, line width=1.55pt"}
        for col, nm in [("r_conf", "Confidence (pooled)"), ("r_conf_cc", "Confidence (class-cond.)"),
                        ("r_agree", "\\method (ours)"), ("r_transductive", "Transductive")]:
            if col not in sn[0]: continue
            sc = np.array([fl(r[col]) for r in sn]); cov, cum, _ = risk_coverage(sc, y)
            idx = np.unique(np.clip((np.linspace(0.02, 1.0, 50) * len(cov)).astype(int) - 1, 0, len(cov) - 1)); ymax = max(ymax, 100 * cum[idx].max())
            series.append((nm, styles[col], " ".join(f"({100*cov[i]:.1f},{100*cum[i]:.2f})" for i in idx)))
        colors = "\\definecolor{figVisual}{RGB}{0,114,178}\\definecolor{figAudio}{RGB}{213,94,0}\\definecolor{figTeal}{RGB}{0,158,115}\\definecolor{figPurple}{RGB}{170,105,170}\\definecolor{figGray}{RGB}{105,112,119}\\definecolor{figInk}{RGB}{35,55,75}"
        common = ("tick label style={font=\\footnotesize,text=figInk}, label style={font=\\footnotesize,text=figInk}, "
                  "axis line style={draw=figInk!50,line width=.55pt}, tick style={draw=figInk!55}, tick align=outside, "
                  "ymajorgrids=true, xmajorgrids=false, grid style={draw=figInk!13,line width=.45pt}, "
                  "title style={font=\\bfseries\\footnotesize,text=figInk,at={(axis description cs:0,1.04)},anchor=south west}")
        fig = "\\begin{figure}[t]\\centering" + colors + "\\resizebox{\\linewidth}{!}{\\begin{tikzpicture}\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.30cm}, scale only axis, height=3.25cm, " + common + "]\n"
        fig += (f"\\nextgroupplot[title={{(a) Selective prediction}}, width=4.8cm, xlabel={{Coverage (\\% retained)}}, ylabel={{Selective risk (\\% wrong)}}, "
                f"xmin=0, xmax=100, xtick={{0,20,...,100}}, ymin=0, ymax={math.ceil(ymax*1.08)}, ytick={{0,10,20,30}}, "
                "legend style={at={(0.5,-0.34)}, anchor=north, font=\\scriptsize, draw=none, fill=none, legend columns=2, "
                "/tikz/every even column/.append style={column sep=0.22cm}, /tikz/every odd column/.append style={column sep=0.08cm}, row sep=1pt}, "
                "legend cell align=left, no markers]\n"
                + "\n".join(f"\\addplot[{st}] coordinates {{{c}}};" for _, st, c in series) + "\n\\legend{" + ", ".join("{" + n + "}" for n, _, _ in series) + "}\n")
        game = np.array([r["game"] for r in sn]); games = sorted(set(game)); rng = np.random.RandomState(1701)
        picks = rng.randint(0, len(games), size=(1000, len(games))); cal_series = []
        for col, nm, style in [("r_conf", "Confidence", "figGray, dashed, mark=square*"), ("r_agree", "\\method", "figVisual, mark=*")]:
            if col not in sn[0]: continue
            score = np.array([fl(r[col]) for r in sn]); pts = []
            for b in range(5):
                m = (score >= b / 5) & (score <= (b + 1) / 5) if b == 4 else (score >= b / 5) & (score < (b + 1) / 5)
                if not m.any(): continue
                ng = np.array([(m & (game == g)).sum() for g in games], float)
                cg = np.array([y[m & (game == g)].sum() for g in games], float)
                den = ng[picks].sum(1); boot = np.divide(cg[picks].sum(1), den, out=np.full(1000, np.nan), where=den > 0)
                yy = float(y[m].mean()); lo, hi = np.nanpercentile(boot, [2.5, 97.5]); err = max(yy - lo, hi - yy)
                pts.append((100 * float(score[m].mean()), 100 * yy, 100 * err))
            cal_series.append((nm, style, " ".join(f"({x:.1f},{yy:.1f}) +- (0,{e:.1f})" for x, yy, e in pts)))
        fig += ("\\nextgroupplot[title={(b) Calibration with 95\\% intervals}, width=4.8cm, xlabel={Predicted correctness (\\%)}, ylabel={Observed correctness (\\%)}, "
                "xmin=0, xmax=100, ymin=0, ymax=100, xtick={0,20,...,100}, ytick={0,20,...,100}, xmajorgrids=true, "
                "legend style={at={(0.5,-0.34)},anchor=north,font=\\scriptsize,draw=none,fill=none,legend columns=2}, legend cell align=left]\n"
                "\\addplot[figInk!45, dotted, line width=1pt, forget plot] coordinates {(0,0) (100,100)};\n"
                + "\n".join(f"\\addplot[{st}, line width=1.25pt, mark size=1.8pt, error bars/.cd, y dir=both, y explicit] coordinates {{{pts}}};" for _, st, pts in cal_series)
                + "\n\\legend{" + ", ".join("{" + n + "}" for n, _, _ in cal_series) + "}\n")
        fig += ("\\end{groupplot}\\end{tikzpicture}}\\caption{Selective reliability on seed-0 test candidates. (a) Tie-aware risk--coverage curves; lower is better. The test-set-wide score-reassignment curve is a transductive comparison. (b) Five-bin reliability diagram. Error bars are 95\\% intervals from 1,000 match bootstraps.}\\label{fig:rc}\\label{fig:reliability}\\end{figure}")
        write("fig_rc.tex", fig)
    # ---------------- rare table
    rare_classes = S.get("rare_classes", [])
    if rare:
        cols = ("rare_recall80", "rare_recall90", "rare_ece", "rare_mean_r", "rare_precision"); A = {k: agg(rare, "selector", k, "selector") for k in cols}
        ns = [int(fl(r["n_rare_correct"])) for r in rare if not isnan(fl(r["n_rare_correct"]))]
        M["nRareMin"], M["nRareMax"] = (str(min(ns)), str(max(ns))) if ns else ("--", "--")
        note = ""
        if sn:
            cnt = {}
            for r in sn:
                if r["class_name"] in rare_classes: cnt[r["class_name"]] = cnt.get(r["class_name"], 0) + 1
            if cnt: note = "Only " + " and ".join(esc(k) for k in cnt) + " produce" + ("s" if len(cnt) == 1 else "") + " candidates at the operating point (" + ", ".join(f"{v} test candidates" for v in cnt.values()) + f" for seed {S.get('export_seed', 0)})"
            M["rareCandNote"] = note
        rn = [esc(c).replace("-$>$", "$\\to$") for c in rare_classes]
        lines = ["\\begin{table}[t]\\centering\\caption{Rare classes (training prior $<1\\%$: " + ", ".join(rn) + "; \\nRareMin--\\nRareMax{} correct rare candidates per seed). Recall of correct rare candidates retained at 80\\% and 90\\% coverage (\\%), rare-class ECE, mean assigned reliability and empirical precision (mean$\\pm$SD over seeds). " + tag + "}\\label{tab:rare}",
                 "\\begin{tabular}{lccccc}\\toprule Selector & Recall@80\\% & Recall@90\\% & ECE & Mean $r$ & Precision \\\\ \\midrule"]
        for k, n in [("Confidence-only", "Confidence-only"), ("Confidence-only class-conditional", "Confidence-only, class-cond."), ("AGREE pooled", "\\method, pooled"), ("AGREE class-conditional", "\\method, class-cond.")]:
            if k in A["rare_recall80"]:
                lines.append(f"{n} & {pm(*A['rare_recall80'][k][:2])} & {pm(*A['rare_recall90'][k][:2])} & {pm(*A['rare_ece'][k][:2], 3, False)} & {pm(*A['rare_mean_r'][k][:2], 2, False)} & {pm(*A['rare_precision'][k][:2], 2, False)} \\\\")
        lines += ["\\bottomrule\\end{tabular}\\end{table}"]; write("tab_rare.tex", "\n".join(lines))
        for k, m in [("Confidence-only", "Conf"), ("Confidence-only class-conditional", "ConfClass"), ("AGREE pooled", "Pool"), ("AGREE class-conditional", "Class")]:
            if k in A["rare_recall80"]: M["rareRecall" + m] = f1(A["rare_recall80"][k][0]) + "\\%"; M["rareRecallNinety" + m] = f1(A["rare_recall90"][k][0]) + "\\%"
        if "AGREE pooled" in A["rare_mean_r"]:
            mr, pr = A["rare_mean_r"]["AGREE pooled"][0], A["rare_precision"]["AGREE pooled"][0]; M["rareMeanR"] = f1(mr, 2, False); M["rarePrecision"] = f1(pr, 2, False)
            M["rareBiasDirection"] = "under-confidence" if mr < pr - 0.02 else ("over-confidence" if mr > pr + 0.02 else "neither direction (pooled reliability and empirical precision agree within two points)")
    # ---------------- ablation table
    if abl:
        F = [r for r in abl if r["ablation"] == "features"]; L = [r for r in abl if r["ablation"] == "lambda"]; R = [r for r in abl if r["ablation"] == "rerank"]; T = [r for r in abl if r["ablation"] == "tau0"]
        lines = ["\\begin{table}[t]\\centering\\caption{Reliability and reranking ablations (mean$\\pm$SD over \\nSeeds{} seeds). Calibration and model selection use match-grouped validation folds.}\\label{tab:ablation}",
                 "\\begin{tabular}{lcc}\\toprule Features & AURC & Risk@80\\% \\\\ \\midrule"]
        fn = {"p_f only": "$p_f$ only", "+A_top": "$p_f$ + $A_{\\mathrm{top}}$", "+A_gap": "$p_f$ + $A_{\\mathrm{gap}}$", "+A_js": "$p_f$ + $A_{\\mathrm{JS}}$", "+A_prod": "$p_f$ + $A_{\\mathrm{prod}}$", "+p_v,p_a": "$p_f$ + $p_v$, $p_a$",
              "p_f + branch scores + agreement (6)": "$z_6$: $p_f$, branch scores + agreement", "agreement + temporal context (11)": "$z_6$ + branch context ($\\pm5$\\,s)", "all (ours, 42: + event-structure context)": "+ event context, shared coeff.", "all 42 + selected coefficients (ours)": "\\textbf{full offline model (ours)}", "all except agreement variables, selected": "full model, no agreement variables", "past-only context, selected": "past-only model (causal)", "visual only, no audio anywhere": "visual only ($p_v$ + context)", "all (ours)": "all (ours)"}
        if F:
            A = {k: agg(F, "setting", k, "setting") for k in ("AURC", "risk80")}; keys = [k for k in fn if k in A["AURC"] and not k.startswith("+")]
            c1 = bold_best([pm(*A["AURC"][k][:2], 3, False) for k in keys], [A["AURC"][k][0] for k in keys], "min"); c2 = bold_best([pm(*A["risk80"][k][:2]) for k in keys], [A["risk80"][k][0] for k in keys], "min")
            for k, a, b in zip(keys, c1, c2): lines.append(f"{fn[k]} & {a} & {b} \\\\")
            for k, m in [("p_f only", "PfOnly"), ("p_f + branch scores + agreement (6)", "AgreeSix"), ("p_f + visual context", "VisCtx"), ("p_f + visual + audio context", "VisAudCtx"), ("all but audio context", "NoAudCtx"), ("agreement + temporal context (11)", "AllShared"), ("all 42 + selected coefficients (ours)", "All"), ("all except agreement variables, selected", "NoAgreement"), ("past-only context, selected", "PastOnly"), ("ours + class-conditional calibration, Eq. (3)", "AllCC"), ("visual only, no audio anywhere", "VisOnly")]:
                if k in A["AURC"]: M["abl" + m] = f1(A["AURC"][k][0], 3, False)
            single = {k: A["AURC"][k][0] for k in A["AURC"] if k.startswith("+A_")}
            if single and "p_f only" in A["AURC"]:
                M["ablMaxSingleDelta"] = f"{max(abs(v - A['AURC']['p_f only'][0]) for v in single.values()):.3f}"
                M["ablBranchDelta"] = f"{A['AURC']['p_f only'][0] - A['AURC']['+p_v,p_a'][0]:.3f}" if "+p_v,p_a" in A["AURC"] else "--"
                M["ablPfOnly"] = f1(A["AURC"]["p_f only"][0], 3, False)
        if L:
            A = agg(L, "setting", "loose", "setting"); keys = [k for k in ["0.0", "0.5", "1.0", "2.0", "4.0"] if k in A]
            mono = all(A[keys[i]][0] > A[keys[i + 1]][0] for i in range(len(keys) - 1)); M["lambdaMonotone"] = "monotone" if mono else "not monotone"
            M["mapLambdaMax"] = f1(A[keys[-1]][0]) if keys else "--"; M["lambdaMax"] = f"{float(keys[-1]):g}" if keys else "--"
            M["lambdaSeries"] = ", ".join(f"{f1(A[k][0])} at $\\lambda={float(k):g}$" for k in keys[1:])
        if R:
            A = agg(R, "setting", "loose", "setting")
            for k, m in [("late fusion (no reranking)", "RerankNone"), ("score-multiset-preserving, candidates >= tau0, 6 features", "RerankZsix"), ("score-multiset-preserving, candidates >= tau0, 11 features", "RerankCand"), ("score-multiset-preserving, all detections >= tau_r, per half, 11 features", "RerankNoCtx"), ("score-multiset-preserving, per half, no agreement variables", "RerankNoAgreement"), ("score-multiset-preserving, per half, past-only context", "RerankPastOnly"), ("score-multiset-preserving, all detections >= tau_r, per half, + event-structure context (ours)", "RerankOurs"), ("score-multiset-preserving, all detections >= tau_r, test-set-wide", "RerankTW"), ("score-compressing p_f r^0.5", "RerankCompress")]:
                if k in A: M["map" + m] = f1(A[k][0])
        if R:
            Ar = agg(R, "setting", "loose", "setting")
            RN = [("late fusion (no reranking)", "late fusion (no reranking)"),
                  ("score-compressing p_f r^0.5", "first reranker: $s=p_f r^{0.5}$ on candidates only"),
                  ("score-multiset-preserving, per half, no agreement variables", "full context, no agreement"),
                  ("score-multiset-preserving, per half, past-only context", "past-only context (causal)"),
                  ("score-multiset-preserving, all detections >= tau_r, per half, + event-structure context (ours)", "\\textbf{full offline \\method}"),
                  ]
            present_r = [(k, n) for k, n in RN if k in Ar]
            if present_r:
                lines.append("\\midrule \\multicolumn{3}{l}{\\emph{Reranking variants (loose mAP, \\%)}} \\\\")
                for k, n in present_r: lines.append(f"{n} & \\multicolumn{{2}}{{c}}{{{pm(*Ar[k][:2])}}} \\\\")
        if T:
            A = {k: agg(T, "setting", k, "setting") for k in ("AURC", "AURC_conf")}
            M["tauZeroSeries"] = "; ".join(f"$\\tau_0={float(k):g}$: {f1(A['AURC'][k][0],3,False)} against {f1(A['AURC_conf'][k][0],3,False)}" for k in ["0.05", "0.1", "0.2"] if k in A["AURC"])
        lines += ["\\bottomrule\\end{tabular}\\end{table}"]; write("tab_ablation.tex", "\n".join(lines))
    # ---------------- degradation + league table (with the undegraded row from the main run)
    LEAGUE_OMIT = ""
    if lg:
        omitted = [LEAGUE_NAMES[L] for L in LEAGUES if L not in {r["league"] for r in lg}]
        if omitted: LEAGUE_OMIT = " (" + " and ".join(omitted) + " omitted: fewer than three test matches)"
    if deg:
        cols = ("loose_late", "loose_qmf", "loose_agree", "AURC_conf", "AURC_agree"); A = {k: agg(deg, "condition", k, "condition") for k in cols}
        if main_rows and sel:
            Am = {k: agg(main_rows, "system", "loose", "system") for k in ("loose",)}["loose"]; As = agg(sel, "selector", "AURC", "selector")
            base = {"loose_late": Am.get("Late fusion (fixed w)"), "loose_qmf": Am.get("QMF (Zhang et al.)", Am.get("Dynamic fusion (QMF-style)")), "loose_agree": Am.get("AGREE rerank (ours)"), "AURC_conf": As.get("Confidence-only (calibrated p_f)"), "AURC_agree": As.get("AGREE, pooled calibration")}
            if all(v is not None for v in base.values()):
                for k in cols: A[k]["none"] = base[k]
        lines = ["\\begin{table}[t]\\centering\\caption{Robustness to test-time degradation (mean$\\pm$SD over \\nSeeds{} seeds; vdrop = fraction of visual frames zeroed, amute = half of the audio muted). Loose mAP (\\%) of late fusion and \\method reranking; AURC of confidence-only against \\method abstention. Leave-one-league-out results are in the text.}\\label{tab:degrade}",
                 "\\begin{tabular}{lcccc}\\toprule Condition & Late fusion & \\method rerank & AURC$_{\\mathrm{conf}}$ & AURC$_{\\method}$ \\\\ \\midrule"]
        for cnd, nm in [("vdrop0.25", "vdrop 0.25"), ("vdrop0.5", "vdrop 0.5"), ("vdrop0.75", "vdrop 0.75"), ("amute0.5", "amute 0.5")]:
            if cnd not in A["loose_late"]: continue
            au = bold_best([pm(*A["AURC_conf"][cnd][:2], 3, False), pm(*A["AURC_agree"][cnd][:2], 3, False)], [A["AURC_conf"][cnd][0], A["AURC_agree"][cnd][0]], "min")
            lines.append(f"{nm} & {pm(*A['loose_late'][cnd][:2])} & {pm(*A['loose_agree'][cnd][:2])} & {au[0]} & {au[1]} \\\\")
        if False:   # league rows folded into prose to save space
            lines.append("\\midrule")
            for r in lg:
                au = bold_best([f1(fl(r["AURC_conf"]), 3, False), f1(fl(r["AURC_agree"]), 3, False)], [fl(r["AURC_conf"]), fl(r["AURC_agree"])], "min")
                lines.append(f"{LEAGUE_SHORT.get(r['league'], esc(r['league']))} ({r['n_test_games']} matches) & {f1(fl(r['loose_late']))} & {f1(fl(r['loose_agree']))} & {au[0]} & {au[1]} \\\\")
        lines += ["\\bottomrule\\end{tabular}\\end{table}"]; write("tab_degrade.tex", "\n".join(lines))
        for cnd, m in [("vdrop0.75", "Worst"), ("vdrop0.25", "Mild"), ("amute0.5", "Amute")]:
            if cnd in A["loose_late"]:
                M["degLate" + m] = f1(A["loose_late"][cnd][0]); M["degAgree" + m] = f1(A["loose_agree"][cnd][0]); M["degQmf" + m] = f1(A["loose_qmf"][cnd][0])
                M["degAurcConf" + m] = f1(A["AURC_conf"][cnd][0], 3, False); M["degAurcAgree" + m] = f1(A["AURC_agree"][cnd][0], 3, False)
        conds = [c for c in ("vdrop0.25", "vdrop0.5", "vdrop0.75", "amute0.5") if c in A["AURC_conf"]]
        M["degWins"] = words(sum(1 for c in conds if A["AURC_agree"][c][0] < A["AURC_conf"][c][0])); M["degN"] = words(len(conds))
        M["degRerankWorse"] = words(sum(1 for c in conds if A["loose_agree"][c][0] < A["loose_late"][c][0])); M["degRerankBetter"] = words(sum(1 for c in conds if A["loose_agree"][c][0] > A["loose_late"][c][0]))
    else: write("tab_degrade.tex", "")
    # ---------------- league macros (table rows are in tab_degrade.tex)
    write("tab_league.tex", "% league rows are the bottom panel of tab_degrade.tex\n")
    if lg:
        M["leagueWins"] = words(sum(1 for r in lg if fl(r["AURC_agree"]) < fl(r["AURC_conf"]))); M["leagueN"] = words(len(lg))
        M["leagueRerankWorse"] = words(sum(1 for r in lg if fl(r["loose_agree"]) < fl(r["loose_late"]))); M["leagueRerankBetter"] = words(sum(1 for r in lg if fl(r["loose_agree"]) > fl(r["loose_late"])))
        eg = min(lg, key=lambda r: fl(r["AURC_agree"]) - fl(r["AURC_conf"])); M["leagueExample"] = f"{LEAGUE_NAMES.get(eg['league'], eg['league'])}: {f1(fl(eg['AURC_conf']),3,False)} against {f1(fl(eg['AURC_agree']),3,False)}"
        M["leagueOmitNote"] = LEAGUE_OMIT
    write("tab_datasets.tex", "% datasets are described in prose (Section 5)\n")
    # ---------------- confidence intervals
    def ci_pts(key, scale=100, d=1):
        if key not in ci: return "--"
        lo, hi = scale * ci[key][1], scale * ci[key][2]
        return "[" + ", ".join((f"$-{-v:.{d}f}$" if v < 0 else f"{v:.{d}f}") for v in (lo, hi)) + "]"
    M["ciMapDiff"] = ci_pts("map_diff"); M["ciAurc"] = ci_pts("aurc_rel", 1).replace(", ", "\\%, ").replace("]", "\\%]") if "aurc_rel" in ci else "--"
    M["ciAurcCC"] = ci_pts("aurc_rel_cc", 1).replace(", ", "\\%, ").replace("]", "\\%]") if "aurc_rel_cc" in ci else "--"
    M["ciAurcDiff"] = ci_pts("aurc_diff", 1, 3); M["ciAurcDiffCC"] = ci_pts("aurc_diff_cc", 1, 3); M["ciAurcDiffCal"] = ci_pts("aurc_diff_cal", 1, 3)
    M["aurcDiffCC"] = f"{ci['aurc_diff_cc'][0]:.3f}" if "aurc_diff_cc" in ci else "--"; M["aurcDiffCal"] = f"{ci['aurc_diff_cal'][0]:.3f}" if "aurc_diff_cal" in ci else "--"
    M["mapDiff"] = (f"$-{-100*ci['map_diff'][0]:.1f}$" if ci["map_diff"][0] < 0 else f"{100*ci['map_diff'][0]:.1f}") if "map_diff" in ci else "--"
    M["aurcRelSeedZero"] = f"{ci['aurc_rel'][0]:.1f}" if "aurc_rel" in ci else "--"; M["aurcRelCCSeedZero"] = f"{ci['aurc_rel_cc'][0]:.1f}" if "aurc_rel_cc" in ci else "--"
    if "map_diff_testwide" in ci: M["ciMapDiffTW"] = ci_pts("map_diff_testwide"); M["mapDiffTW"] = f"{100*ci['map_diff_testwide'][0]:.1f}"
    # increment of the event-structure context over the same reranker with the 11 reliability features only (seed 0, paired over matches);
    # defined as placeholders first so a build never breaks while the seed-0 bootstrap re-run is still in flight
    M.setdefault("ciMapDiffCtx", "--"); M.setdefault("mapDiffCtx", "--"); M.setdefault("ciUnshownDiffCtx", "--"); M.setdefault("unshownDiffCtx", "--")
    # increment of the event-structure context over the same reranker with the 11 reliability features only (seed 0, paired over matches)
    if "mapAgree" in M and "mapNetVLAD" in M:
        try: M["unshownGapNote"] = f"{float(M['mapAgree']) - float(M['mapNetVLAD']):.1f}-point"
        except Exception: M["unshownGapNote"] = "clear"
    if "map_diff_ctx" in ci: M["ciMapDiffCtx"] = ci_pts("map_diff_ctx"); M["mapDiffCtx"] = f"{100*ci['map_diff_ctx'][0]:.1f}"
    if "unshown_diff_ctx" in ci: M["ciUnshownDiffCtx"] = ci_pts("unshown_diff_ctx"); M["unshownDiffCtx"] = f"{100*ci['unshown_diff_ctx'][0]:.1f}"
    rv = S.get("rare_val_by_seed", {})
    if rv:
        pen = [v.get("Penalty", 0) for v in rv.values()]; M["rareValMin"], M["rareValMax"] = str(min(pen)), str(max(pen)); M["rareValBelow"] = words(sum(1 for x in pen if x < 30)); M["rareValN"] = words(len(pen))
    t0s = S.get("tau0_by_seed", {}); 
    if t0s:
        vals = list(t0s.values()); m0 = mode(vals); oth = sorted(set(v for v in vals if v != m0)); M["tauZeroOther"] = (", ".join(f"{v:g}" for v in oth) if oth else "--")
    rev = os.path.join(RESULTS, f"revision_s{S.get('export_seed', 0)}.json")
    if os.path.exists(rev):
        R = json.load(open(rev)); cp = R.get("ci_pooled", {})
        def pts(v, scale=1, d=1): return "[" + ", ".join(f"{scale*x:.{d}f}" for x in v[1:3]) + "]"
        # the recommended estimator (pooled, cross-fitted) is r_agree in the final run, so its CIs come from summary.json; the
        # revision-study values are used only if the run lacks them
        if "aurc_rel" in ci: M["ciAurcPool"] = M["ciAurc"]; M["aurcRelPoolSeedZero"] = M["aurcRelSeedZero"]
        elif "aurc_rel_pool_vs_conf" in cp: M["ciAurcPool"] = pts(cp["aurc_rel_pool_vs_conf"]).replace(", ", "\\%, ").replace("]", "\\%]"); M["aurcRelPoolSeedZero"] = f"{cp['aurc_rel_pool_vs_conf'][0]:.1f}"
        if "aurc_rel_cc" in ci: M["ciAurcPoolCC"] = M["ciAurcCC"]; M["aurcRelPoolCCSeedZero"] = M["aurcRelCCSeedZero"]
        elif "aurc_rel_pool_vs_confcc" in cp: M["ciAurcPoolCC"] = pts(cp["aurc_rel_pool_vs_confcc"]).replace(", ", "\\%, ").replace("]", "\\%]"); M["aurcRelPoolCCSeedZero"] = f"{cp['aurc_rel_pool_vs_confcc'][0]:.1f}"
        M["ciAurcDiffPoolCC"] = M["ciAurcDiffCC"] if "aurc_diff_cc" in ci else (pts(cp["aurc_diff_pool_vs_confcc"], 1, 3) if "aurc_diff_pool_vs_confcc" in cp else "--")
        M["ciAurcDiffPool"] = M["ciAurcDiff"] if "aurc_diff" in ci else (pts(cp["aurc_diff_pool_vs_conf"], 1, 3) if "aurc_diff_pool_vs_conf" in cp else "--")
        if "first_reranker_frac_below_tau0" in R: M["firstRerankFracBelow"] = f1(R["first_reranker_frac_below_tau0"])
        for k, m in [("rerank_perhalf[all 11 (ours)]", "mapRerankPerHalf"), ("rerank_testwide[p_f + visual context]", "mapRerankVisCtx"), ("rerank_testwide[visual only (p_v + visual context)]", "mapRerankVisOnly"), ("first_reranker_ordering_rank_preserving", "mapFirstOrderingRP"), ("rerank_testwide[all 11 (ours)]", "mapRerankSeedZero"), ("late_fusion", "mapLateSeedZero")]:
            if k in R: M[m] = f1(R[k]["loose"]); M["t" + m] = f1(R[k]["tight"]); M["u" + m] = f1(R[k]["unshown"])
    # ---------------- RQ3 study (in-sample vs cross-fitted calibration on the rare class), averaged over the seeds present
    import glob as _glob
    rq = [json.load(open(p)) for p in sorted(_glob.glob(os.path.join(RESULTS, "rq3_s*.json")))]
    if rq:
        def mean(key, sub=None): return float(np.mean([r[key][sub] if sub else r[key] for r in rq]))
        M["rqAucIn"] = f"{mean('rare_val_auc_insample'):.2f}"; M["rqAucOof"] = f"{mean('rare_val_auc_oof'):.2f}"; M["rqSeeds"] = words(len(rq))
        M["rqValRange"] = f"{min(r['n_rare_val'] for r in rq)}--{max(r['n_rare_val'] for r in rq)}"
        for k, m in [("iso_pooled_insample", "PoolIn"), ("iso_classcond_insample", "CCIn"), ("iso_pooled_crossfit", "PoolCF"), ("iso_classcond_crossfit", "CCCF"), ("offset_kappa5_crossfit", "OffCF")]:
            M["rqRecall" + m] = f1(mean(k, "rare_recall80")) + "\\%"; M["rqAurc" + m] = f1(mean(k, "AURC"), 3, False); M["rqMeanR" + m] = f1(mean(k, "rare_mean_r"), 2, False)
        M["rqPrecision"] = f1(mean("iso_pooled_insample", "rare_precision"), 2, False)
        wins = sum(1 for r in rq if r["iso_classcond_crossfit"]["rare_recall80"] >= r["iso_pooled_crossfit"]["rare_recall80"]); M["rqCCWinsCF"] = words(wins)
        wins = sum(1 for r in rq if r["iso_classcond_insample"]["rare_recall80"] < r["iso_pooled_insample"]["rare_recall80"]); M["rqCCLosesIn"] = words(wins)
    # ---------------- write macros (every name gets a default so the manuscript always compiles)
    defaults = ["nSeeds", "nTestGames", "nBoot", "nBootMap", "nCandTest", "nCandValid", "candPrecision", "tauZero", "tauZeroNote", "fOneValid", "omegaVal", "omegaValNote", "lambdaVal", "lambdaValNote", "tauQmf", "tauQmfNote",
                "mapVisual", "mapAudio", "mapLate", "mapJoint", "mapMeanEns", "mapEns", "mapQmf", "mapAgree", "tmapVisual", "tmapLate", "tmapAgree", "unshownVisual", "unshownLate", "unshownAgree", "bypassDelta", "ciMapDiff", "mapDiff",
                "aurcConf", "aurcAgree", "aurcPool", "aurcEns", "aurcQmf", "aurcPfClass", "aurcRel", "aurcRelCC", "aurcRelCal", "ciAurc", "ciAurcCC", "ciAurcDiff", "ciAurcDiffCC", "ciAurcDiffCal", "aurcDiffCC", "aurcDiffCal",
                "riskEightyConf", "riskEightyAgree", "riskNinetyConf", "riskNinetyAgree", "riskEightyPfClass", "eceConf", "eceAgree", "eceEns", "brierConf", "brierAgree", "tieSpanConf", "tieSpanAgree",
                "rhoTop", "rhoProd", "rhoJs", "rhoGap", "rhoPv", "rhoPa", "rhoPf", "rhoTopVis", "rhoTopUns", "rhoProdVis", "rhoProdUns", "rhoPfVis", "rhoPfUns", "rhoPvVis", "rhoPvUns", "rhoPaVis", "rhoPaUns", "rhoJsVis", "rhoJsUns", "rhoGapVis", "rhoGapUns", "phiAV", "phiVV", "phiAVpos", "phiVVpos", "nUnsel", "errRateV", "errRateA", "errRateVV", "missRateV", "missRateA", "missRateVV",
                "fracNearVisible", "fracNearUnshown", "fracNearNone", "binLow", "binHigh", "binSdMin", "binSdMax", "binDip", "binDipDecile",
                "rareRecallConf", "rareRecallConfClass", "rareRecallPool", "rareRecallClass", "rareRecallNinetyConf", "rareRecallNinetyClass", "rareRecallNinetyPool", "rareRecallNinetyConfClass", "rareMeanR", "rarePrecision", "rareBiasDirection", "rareCandNote", "nRareMin", "nRareMax",
                "ablMaxSingleDelta", "ablBranchDelta", "ablPfOnly", "lambdaMonotone", "mapLambdaMax", "lambdaMax", "lambdaSeries", "ablAgreeSix", "ablVisCtx", "ablVisAudCtx", "ablNoAudCtx", "ablAll", "ablAllShared", "ablNoAgreement", "ablPastOnly", "ablVisOnly", "mapRerankNone", "mapRerankZsix", "mapRerankCand", "mapRerankNoCtx", "mapRerankNoAgreement", "mapRerankPastOnly", "mapRerankOurs", "mapRerankCompress", "tauR", "tauRNote", "combinerNote", "rerankCombinerNote", "mapAgreeCompress", "tmapAgreeCompress", "unshownAgreeCompress",
                "degLateWorst", "degAgreeWorst", "degQmfWorst", "degAurcConfWorst", "degAurcAgreeWorst", "degLateMild", "degAgreeMild", "degAurcConfMild", "degAurcAgreeMild", "degAurcConfAmute", "degAurcAgreeAmute", "degWins", "degN", "degRerankWorse",
                "leagueWins", "leagueN", "leagueRerankWorse", "leagueRerankBetter", "degRerankBetter", "leagueExample", "leagueOmitNote", "tauZeroSeries", "qmfVsLate", "aurcAgreeCC", "ciMapDiffTW", "rqAucIn", "rqAucOof", "rqSeeds", "rqValRange", "rqRecallPoolIn", "rqRecallCCIn", "rqRecallPoolCF", "rqRecallCCCF", "rqRecallOffCF", "rqAurcPoolIn", "rqAurcCCIn", "rqAurcPoolCF", "rqAurcCCCF", "rqAurcOffCF", "rqMeanRPoolIn", "rqMeanRCCIn", "rqMeanRPoolCF", "rqMeanRCCCF", "rqMeanROffCF", "rqPrecision", "rqCCWinsCF", "rqCCLosesIn", "mapDiffTW", "rareValMin", "rareValMax", "rareValBelow", "rareValN", "tauZeroOther", "ablAllCC", "mapRerankTW", "mapAgreeTW", "tmapAgreeTW", "unshownAgreeTW", "riskEightyAgreeCC", "eceAgreeCC", "phiAVneg", "phiVVneg", "aurcRelSeedZero", "aurcRelCCSeedZero", "ciAurcPool", "ciAurcPoolCC", "ciAurcDiffPool", "ciAurcDiffPoolCC", "aurcRelPoolSeedZero", "aurcRelPoolCCSeedZero", "firstRerankFracBelow", "mapRerankPerHalf", "tmapRerankPerHalf", "umapRerankPerHalf", "mapRerankVisCtx", "mapRerankVisOnly", "mapFirstOrderingRP", "mapRerankSeedZero", "mapLateSeedZero", "nCandRange"]
    for k in defaults: M.setdefault(k, (tag + "--") if stub else "--")
    M.pop("relBins", None); M.pop("relPanel", None)
    hdr = "% generated by src/render.py -- do not edit (run: python src/render.py)\n"
    write("macros.tex", hdr + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
    for f in ["tab_main", "tab_selective", "tab_diagnostic", "fig_reliability", "fig_rc", "tab_rare", "tab_ablation", "tab_degrade", "tab_league", "tab_datasets"]:
        if not os.path.exists(os.path.join(GEN, f + ".tex")): write(f + ".tex", f"% {f}: not produced in this run\n")
    print("rendered", len(M), "macros to", GEN)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--stub", action="store_true"); a = p.parse_args(); render(stub=a.stub)
