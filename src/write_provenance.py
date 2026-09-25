"""Write a SHA-256 manifest linking code, results, paper sources, plus the PDF.

Large checkpoints and predictions are excluded by default. Pass --full to hash them as well.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import importlib.metadata

from common import ROOT, RESULTS


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def git_revision(repo):
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


def git_dirty(repo):
    try:
        return bool(subprocess.check_output(["git", "-C", repo, "status", "--porcelain"], text=True).strip())
    except Exception:
        return None


def main(full=False):
    project = os.path.dirname(ROOT); manuscript = os.path.join(project, "manuscript")
    roots = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tests"), os.path.join(manuscript, "generated"),
             os.path.join(manuscript, "sections"), os.path.join(manuscript, "images")]
    single = [os.path.join(ROOT, "README.md"), os.path.join(ROOT, "STATUS.md"), os.path.join(ROOT, "REPRODUCE.md"), os.path.join(ROOT, "QMF_PORT.md"),
              os.path.join(ROOT, "requirements-lock.txt"), os.path.join(ROOT, "CITATION.cff"), os.path.join(ROOT, "run_all.sh"),
              os.path.join(ROOT, "Makefile"), os.path.join(manuscript, "main.tex"),
              os.path.join(manuscript, "references.bib"), os.path.join(manuscript, "main.pdf"),
              os.path.join(manuscript, "figure1-framework-editable.pptx")]
    single += [os.path.join(project, name) for name in
               ("STATUS.md", "codex-review.md", "codex-citation.md", "revision-execution-report.md")]
    canonical = ["main_results.csv", "selective.csv", "diagnostic.csv", "ablation.csv", "rare.csv",
                 "degradation.csv", "league.csv", "sn_agree_test.csv", "sn_agree_valid.csv", "summary.json",
                 "baselines_official.json", "reported_sota.json"]
    single += [os.path.join(RESULTS, n) for n in canonical]
    if full:
        roots += [os.path.join(ROOT, "models"), os.path.join(RESULTS, "preds"),
                  os.path.join(ROOT, "baselines", "eval_stage"),
                  os.path.join(ROOT, "baselines", "server_sync", "CALF", "models"),
                  os.path.join(ROOT, "baselines", "server_sync", "CALF", "outputs"),
                  os.path.join(ROOT, "baselines", "server_sync", "TemporallyAwarePooling", "models")]
        calf = os.path.join(ROOT, "baselines", "server_sync", "CALF")
        if os.path.isdir(calf):
            roots += [os.path.join(calf, name) for name in sorted(os.listdir(calf)) if name.startswith("outputs_")]
    files = [p for p in single if os.path.isfile(p)]
    for base in roots:
        if not os.path.isdir(base): continue
        for dp, dirs, names in os.walk(base, followlinks=base.endswith("eval_stage")):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            if not full and (dp.startswith(os.path.join(RESULTS, "preds")) or dp.startswith(os.path.join(ROOT, "models"))): continue
            files.extend(os.path.join(dp, n) for n in names
                         if not n.startswith(".") and not n.endswith((".pyc", ".pyo"))
                         and os.path.isfile(os.path.join(dp, n)))
    files = sorted(set(files)); manifest = {os.path.relpath(p, project): digest(p) for p in files}
    outdir = os.path.join(ROOT, "provenance"); os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "manifest.sha256"), "w") as f:
        for p, h in manifest.items(): f.write(f"{h}  {p}\n")
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "SoccerNet", "pandas", "matplotlib"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = "not installed"
    summary_path = os.path.join(RESULTS, "summary.json")
    summary = json.load(open(summary_path)) if os.path.isfile(summary_path) else {}
    meta = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "git_revision": git_revision(project),
            "python": sys.version, "platform": platform.platform(), "full_manifest": full, "git_dirty": git_dirty(project),
            "result_dir": os.path.relpath(RESULTS, project), "validation_folds": "GroupKFold by match",
            "aurc_ties": "expected risk under uniform ordering within equal-score blocks", "packages": packages,
            "command": " ".join(sys.argv), "dataset": "SoccerNet-v2", "split": "300 train / 100 validation / 100 test matches",
            "seeds": summary.get("seeds", []), "n_valid_games": summary.get("n_valid_games"),
            "n_test_games": summary.get("n_test_games"), "analysis_started_utc": summary.get("analysis_started_utc"),
            "analysis_finished_utc": summary.get("analysis_finished_utc"), "analysis_command": summary.get("analysis_command")}
    with open(os.path.join(outdir, "run.json"), "w") as f: json.dump(meta, f, indent=2)
    print(f"wrote {len(manifest)} hashes to {os.path.relpath(outdir, ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--full", action="store_true"); a = p.parse_args(); main(a.full)
