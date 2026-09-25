"""Add the validation-calibrated transductive selector to completed per-seed outputs.

This postprocessor reuses the selector/ranker configurations recorded by a completed
``run_pipeline.py`` process. It avoids repeating model selection when only the
test-set-wide score-reassignment diagnostic is newly requested.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS
from analysis import GameSet, build_candidates, late_fusion, load_logits, rerank_score_multiset, sigmoid
from run_pipeline import (FULL_R, attach_detection_scores, available_games,
                          fit_reliability, grouped_splits, selective_table)


NAME = "AGREE, transductive score reassignment"


def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def by_seed(mapping, seed):
    return mapping[str(seed)] if str(seed) in mapping else mapping[seed]


def main(seed, result_dir):
    out = os.path.join(RESULTS, result_dir)
    suffix = f"_s{seed}"
    summary_path = os.path.join(out, f"summary{suffix}.json")
    summary = json.load(open(summary_path))
    omega = float(by_seed(summary["omega_by_seed"], seed))
    tau0 = float(by_seed(summary["tau0_by_seed"], seed))
    rcfg = by_seed(summary["rerank_by_seed"], seed)

    tags = [f"visual_s{seed}", f"audio_s{seed}"]
    valid_games = available_games("valid", tags)
    test_games = available_games("test", tags)
    gv, gt = GameSet(valid_games), GameSet(test_games)

    def branch_arrays(gs):
        zv = [load_logits(f"visual_s{seed}", game, half) for game in gs.games for half in (1, 2)]
        za = [load_logits(f"audio_s{seed}", game, half) for game in gs.games for half in (1, 2)]
        return [sigmoid(x) for x in zv], [sigmoid(x) for x in za]

    pv_v, pa_v = branch_arrays(gv); pv_t, pa_t = branch_arrays(gt)
    pf_v = late_fusion(pv_v, pa_v, omega); pf_t = late_fusion(pv_t, pa_t, omega)
    rows_v = build_candidates(gv, pf_v, pv_v, pa_v, thr=tau0)
    rows_t = build_candidates(gt, pf_t, pv_t, pa_t, thr=tau0)
    rows_v_r = build_candidates(gv, pf_v, pv_v, pa_v, thr=float(rcfg["thr"]))
    rows_t_r = build_candidates(gt, pf_t, pv_t, pa_t, thr=float(rcfg["thr"]))

    kwargs = {"class_inter": bool(rcfg["class_inter"]), "C": float(rcfg["C"])}
    ranker = fit_reliability(rows_v_r, FULL_R, class_cond=True, cross_fit=False, **kwargs)
    g_v_oof = np.zeros(len(rows_v_r))
    for train, test in grouped_splits(rows_v_r):
        fold = fit_reliability([rows_v_r[i] for i in train], FULL_R, class_cond=True,
                               cross_fit=False, **kwargs)
        g_v_oof[test] = fold.raw([rows_v_r[i] for i in test])

    d_v = rerank_score_multiset(gv, pf_v, rows_v_r, g_v_oof)
    d_t = rerank_score_multiset(gt, pf_t, rows_t_r, ranker.raw(rows_t_r))
    attach_detection_scores(rows_v, gv, d_v, "p_transductive")
    attach_detection_scores(rows_t, gt, d_t, "p_transductive")
    selector = fit_reliability(rows_v, ["p_transductive"], class_cond=False)
    scores = selector(rows_t)

    selective_path = os.path.join(out, f"selective{suffix}.csv")
    selective, fields = read_csv(selective_path)
    selective = [row for row in selective if row.get("selector") != NAME]
    row = {"seed": seed, **selective_table(rows_t, scores, NAME)}
    write_csv(selective_path, selective + [row], fields)

    if seed == 0:
        candidate_path = os.path.join(out, "sn_agree_test_s0.csv")
        exported, candidate_fields = read_csv(candidate_path)
        if len(exported) != len(rows_t):
            raise RuntimeError(f"candidate count mismatch: {len(exported)} != {len(rows_t)}")
        for old, new, score in zip(exported, rows_t, scores):
            old_key = (old["game"], int(old["half"]), int(old["frame"]), int(old["cls"]))
            new_key = (new["game"], int(new["half"]), int(new["frame"]), int(new["cls"]))
            if old_key != new_key:
                raise RuntimeError(f"candidate order mismatch: {old_key} != {new_key}")
            old["r_transductive"] = float(score)
        if "r_transductive" not in candidate_fields: candidate_fields.append("r_transductive")
        write_csv(candidate_path, exported, candidate_fields)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dir", default="revision_final")
    args = parser.parse_args()
    main(args.seed, args.dir)
