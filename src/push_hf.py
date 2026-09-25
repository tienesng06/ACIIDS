"""Push compact release files to a configured Hugging Face dataset repository.

Authentication uses the standard HF_TOKEN environment variable. Example:
HF_TOKEN=... python src/push_hf.py --repo organization/anonymous-artifact --what all
"""
import os, argparse
from huggingface_hub import HfApi
HERE = os.path.dirname(os.path.abspath(__file__)); ART = os.path.dirname(HERE); PROJ = os.path.dirname(ART)

def main(a):
    token = os.environ.get("HF_TOKEN")
    if not token: raise SystemExit("HF_TOKEN is required; see https://huggingface.co/docs/huggingface_hub/authentication")
    repo = a.repo or os.environ.get("HF_REPO_ID")
    if not repo: raise SystemExit("set --repo or HF_REPO_ID")
    api = HfApi(token=token)
    what = set(a.what) if "all" not in a.what else {"manuscript", "results", "src"}
    if "manuscript" in what:
        info = api.upload_folder(folder_path=os.path.join(PROJ, "manuscript"), path_in_repo="manuscript", repo_id=repo, repo_type="dataset",
                                 allow_patterns=["main.pdf", "main.tex", "references.bib", "sections/*.tex", "generated/*.tex", "*.pptx", "images/*.png", "images/*.md"],
                                 commit_message=a.message or "manuscript update")
        print("manuscript ->", info.commit_url, flush=True)
    if "results" in what:
        info = api.upload_folder(folder_path=os.path.join(ART, "results"), path_in_repo="results", repo_id=repo, repo_type="dataset",
                                 allow_patterns=["*.csv", "*.json"], ignore_patterns=["preds/**", "v[0-9]*/**"], commit_message=a.message or "results update")
        print("results ->", info.commit_url, flush=True)
    if "src" in what:
        info = api.upload_folder(folder_path=os.path.join(ART, "src"), path_in_repo="src", repo_id=repo, repo_type="dataset",
                                 allow_patterns=["*.py", "*.sh"], ignore_patterns=["__pycache__/**"], commit_message=a.message or "src update")
        print("src ->", info.commit_url, flush=True)
        for f in ("STATUS.md", "PLAN.md"):
            api.upload_file(path_or_fileobj=os.path.join(PROJ, f), path_in_repo=f, repo_id=repo, repo_type="dataset", commit_message=f"{f} update")
        print("STATUS.md, PLAN.md pushed", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--repo", default=""); p.add_argument("--what", nargs="+", default=["all"]); p.add_argument("--message", default=""); main(p.parse_args())
