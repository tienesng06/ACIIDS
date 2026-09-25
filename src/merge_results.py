"""Merge per-seed outputs of run_pipeline.py (--out-tag _s<k>) into the canonical results/*.csv and summary.json.
Usage: python src/merge_results.py [--seeds 0 1 2 3 4]
"""
import os, sys, csv, json, argparse, glob
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS

def read(p):
    with open(p) as f: return list(csv.DictReader(f))

def main(seeds, sub=""):
    global RESULTS
    if sub: RESULTS = os.path.join(RESULTS, sub)
    for name in ["main_results", "selective", "diagnostic", "ablation", "rare", "degradation"]:
        rows, fields = [], None
        for s in seeds:
            p = os.path.join(RESULTS, f"{name}_s{s}.csv")
            if not os.path.exists(p): print("missing", p); continue
            with open(p) as f:
                rd = csv.DictReader(f); fields = fields or rd.fieldnames; rows += list(rd)
        if rows:
            with open(os.path.join(RESULTS, f"{name}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
            print(name, len(rows), "rows")
    s0 = seeds[0]
    for name in ["league", "sn_agree_test", "sn_agree_valid"]:
        p = os.path.join(RESULTS, f"{name}_s{s0}.csv")
        if os.path.exists(p): os.replace(p, os.path.join(RESULTS, f"{name}.csv")); print("moved", name)
    summ = {}
    for s in seeds:
        p = os.path.join(RESULTS, f"summary_s{s}.json")
        if not os.path.exists(p): continue
        S = json.load(open(p))
        for k in ["tau0_by_seed", "omega_by_seed", "tau_by_seed", "lambda_by_seed", "combiner_by_seed", "rerank_by_seed", "rare_val_by_seed"]:
            summ.setdefault(k, {}).update(S.get(k, {}))
        if s == s0: summ.update({k: v for k, v in S.items() if k not in ("seeds",)})
    if summ:
        summ["seeds"] = seeds; json.dump(summ, open(os.path.join(RESULTS, "summary.json"), "w"), indent=2); print("summary.json written")

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); a.add_argument("--dir", default=""); x = a.parse_args(); main(x.seeds, x.dir)
