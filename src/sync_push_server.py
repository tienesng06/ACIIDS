"""Pull baseline outputs from a configured GPU server, then upload them to a configured HF repository.
Authentication uses HF_TOKEN. The server endpoint uses --remote or SN_SYNC_REMOTE.
Usage: python src/sync_push_server.py --remote HOST:/path --repo ORG/REPO [--once]
Stops after a final sync+push once the server has written baselines/logs/ALL_DONE (or after --max-hours).
"""
import os, sys, time, subprocess, argparse, datetime
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT
LOCAL = os.path.join(ROOT, "baselines", "server_sync"); REMOTE = ""
RSYNC = ["rsync", "-a", "--mkpath", "--timeout=120", "--include=*/", "--include=logs/**", "--include=*/models/**", "--include=CALF/outputs/**", "--include=CALF/outputs_*/**", "--include=*/src/**", "--include=*.sh", "--exclude=*"]

def log(*a): print(datetime.datetime.now().strftime("%H:%M:%S"), *a, flush=True)

def sync():
    os.makedirs(LOCAL, exist_ok=True)
    r = subprocess.run(RSYNC + [REMOTE, LOCAL + "/"], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0: log("rsync failed:", r.stderr.strip()[:300]); return False
    return True

def push(note, repo):
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token: raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)
    info = api.upload_folder(folder_path=LOCAL, path_in_repo="baselines/" + os.path.basename(LOCAL), repo_id=repo, repo_type="dataset",
                             commit_message=f"server sync {datetime.datetime.now():%Y-%m-%d %H:%M} ({note})", ignore_patterns=["*.tmp"])
    return getattr(info, "commit_url", str(info))

def main(a):
    global REMOTE, LOCAL
    REMOTE = a.remote or os.environ.get("SN_SYNC_REMOTE", "")
    repo = a.repo or os.environ.get("HF_REPO_ID", "")
    if not REMOTE: raise SystemExit("set --remote or SN_SYNC_REMOTE")
    if not repo: raise SystemExit("set --repo or HF_REPO_ID")
    label = REMOTE.split(":", 1)[0].replace("/", "_"); LOCAL = os.path.join(ROOT, "baselines", f"server_sync_{label}")
    log(f"mirroring {REMOTE} -> {LOCAL}")
    t0 = time.time(); n = 0
    while True:
        n += 1; ok = sync()
        done = ok and os.path.exists(os.path.join(LOCAL, "logs", "ALL_DONE"))
        if ok:
            try: url = push("final" if done else f"round {n}", repo); log(f"round {n}: synced and pushed -> {url}")
            except Exception as e: log(f"round {n}: push failed: {str(e)[:300]}")
        if done: log("server reported ALL_DONE; final push made; exiting"); break
        if a.once or time.time() - t0 > a.max_hours * 3600: log("stopping (once/max-hours)"); break
        time.sleep(a.interval)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--interval", type=int, default=600); p.add_argument("--max-hours", type=float, default=14); p.add_argument("--once", action="store_true"); p.add_argument("--remote", default=""); p.add_argument("--repo", default=""); main(p.parse_args())
