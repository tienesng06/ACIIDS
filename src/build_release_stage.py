"""Build a coherent reviewer snapshot without redistributing licensed source data.

The default package includes code, tests, compact results, paper sources, the PDF,
the editable Figure 1 slide, plus provenance. Models may be included explicitly.
Large prediction arrays remain external licensed derivatives; a full provenance
manifest records them when ``write_provenance.py --full`` is used.
"""
import argparse
import os
import shutil
import tempfile

from common import ROOT, RESULTS


PROJECT = os.path.dirname(ROOT)
MANUSCRIPT = os.path.join(PROJECT, "manuscript")


def copy_file(src, dst_root, rel=None):
    if not os.path.isfile(src):
        return
    rel = rel or os.path.relpath(src, PROJECT)
    dst = os.path.join(dst_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src, dst_root, rel=None, suffixes=None, hardlink=False, followlinks=False):
    if not os.path.isdir(src):
        return
    base_rel = rel or os.path.relpath(src, PROJECT)
    for dp, dirs, names in os.walk(src, followlinks=followlinks):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in names:
            if name.startswith(".") or name.endswith((".pyc", ".pyo")):
                continue
            if suffixes and not name.endswith(suffixes):
                continue
            path = os.path.join(dp, name)
            out_rel = os.path.join(base_rel, os.path.relpath(path, src))
            if hardlink:
                dst = os.path.join(dst_root, out_rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    os.link(path, dst)
                except OSError:
                    shutil.copy2(path, dst)
            else:
                copy_file(path, dst_root, out_rel)


def build(output, include_models=False, include_predictions=False, include_baselines=False):
    if os.path.exists(output):
        raise FileExistsError(f"refusing to overwrite existing snapshot: {output}")
    parent = os.path.dirname(output)
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".release-stage-", dir=parent)
    try:
        for name in ("README.md", "STATUS.md", "REPRODUCE.md", "QMF_PORT.md", "requirements-lock.txt", "CITATION.cff", "Makefile", "run_all.sh"):
            copy_file(os.path.join(ROOT, name), tmp)
        copy_file(os.path.join(ROOT, "README.md"), tmp, "README.md")
        for name in ("STATUS.md", "codex-review.md", "codex-citation.md", "revision-execution-report.md"):
            copy_file(os.path.join(PROJECT, name), tmp, name)
        copy_tree(os.path.join(ROOT, "src"), tmp)
        copy_tree(os.path.join(ROOT, "tests"), tmp)
        copy_tree(os.path.join(ROOT, "provenance"), tmp)
        canonical = ("main_results.csv", "selective.csv", "diagnostic.csv", "ablation.csv", "rare.csv",
                     "degradation.csv", "league.csv", "sn_agree_test.csv", "sn_agree_valid.csv", "summary.json",
                     "baselines_official.json", "reported_sota.json")
        for name in canonical:
            copy_file(os.path.join(RESULTS, name), tmp)
        for name in ("main.tex", "references.bib", "llncs.cls", "splncs04.bst", "main.pdf",
                     "figure1-framework-editable.pptx", "fig1_editable.pptx"):
            copy_file(os.path.join(MANUSCRIPT, name), tmp)
        copy_tree(os.path.join(MANUSCRIPT, "sections"), tmp)
        copy_tree(os.path.join(MANUSCRIPT, "generated"), tmp)
        copy_tree(os.path.join(MANUSCRIPT, "images"), tmp,
                  suffixes=(".py", ".png", ".json", ".md"))
        if include_models:
            copy_tree(os.path.join(ROOT, "models"), tmp, hardlink=True)
        if include_predictions:
            copy_tree(os.path.join(RESULTS, "preds"), tmp, hardlink=True)
        if include_baselines:
            copy_tree(os.path.join(ROOT, "baselines", "eval_stage"), tmp,
                      hardlink=True, followlinks=True)
            sync = os.path.join(ROOT, "baselines", "server_sync")
            copy_tree(os.path.join(sync, "CALF", "models"), tmp, hardlink=True)
            calf = os.path.join(sync, "CALF")
            if os.path.isdir(calf):
                for name in sorted(os.listdir(calf)):
                    if name.startswith("outputs"):
                        copy_tree(os.path.join(calf, name), tmp, hardlink=True)
            copy_tree(os.path.join(sync, "TemporallyAwarePooling", "models"), tmp, hardlink=True)
        os.replace(tmp, output)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(os.path.relpath(output, PROJECT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(ROOT, "hf_upload", "publish-root", "releases", "aciids-2027-review-20260907"))
    parser.add_argument("--include-models", action="store_true")
    parser.add_argument("--include-predictions", action="store_true")
    parser.add_argument("--include-baselines", action="store_true")
    args = parser.parse_args()
    build(os.path.abspath(args.output), args.include_models, args.include_predictions, args.include_baselines)
