"""Score the official NetVLAD++ / CALF codebases' test-split outputs (results_spotting.json per game, produced by the
authors' code retrained on our ResNET_TF2_PCA512 features) with the official SoccerNet evaluation at both tolerances.
Usage: python src/eval_official_baselines.py --pred-root <dir with <model>/<league>/<season>/<game>/results_spotting.json> --name NetVLAD++ --runs run_0 run_1 ...
Appends to results/baselines_official.json.
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, RESULTS
from SoccerNet.Evaluation.ActionSpotting import evaluate

def main(a):
    out_p = os.path.join(RESULTS, "baselines_official.json"); out = json.load(open(out_p)) if os.path.exists(out_p) else {"rows": []}
    rows = []
    for run in a.runs:
        d = os.path.join(a.pred_root, run)
        if not os.path.isdir(d): print("missing", d); continue
        r = {}
        for metric in ("loose", "tight"):
            res = evaluate(SoccerNet_path=DATA, Predictions_path=d, split="test", version=2, prediction_file=a.prediction_file, metric=metric)
            r[metric] = float(res["a_mAP"]); r[metric + "_visible"] = float(res["a_mAP_visible"]); r[metric + "_unshown"] = float(res["a_mAP_unshown"])
        r["run"] = run; rows.append(r); print(run, r, flush=True)
    if rows:
        agg = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in rows[0] if k != "run"}
        out["rows"] = [x for x in out["rows"] if x["name"] != a.name] + [{"name": a.name, "cite": a.cite, "n_runs": len(rows), "runs": rows, **agg, "note": a.note}]
        json.dump(out, open(out_p, "w"), indent=2); print("written", out_p)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--pred-root", required=True); p.add_argument("--name", required=True); p.add_argument("--cite", required=True)
    p.add_argument("--runs", nargs="+", required=True); p.add_argument("--prediction-file", default="results_spotting.json"); p.add_argument("--note", default="official code retrained on our ResNet-152 PCA-512 features"); main(p.parse_args())
