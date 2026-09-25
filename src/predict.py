"""Write per-frame logits for a trained branch on a split, optionally with test-time degradation.
Usage: python src/predict.py --tag visual_s0 --split test [--visual-dropout 0.5] [--audio-mute 0.5] [--out-tag name]
"""
import os, sys, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from train_branch import build_model, QMFNet

def degrade(x, modality, args, rng):
    x = x.copy()
    if modality in ("visual", "joint", "qmf") and args.visual_dropout > 0:
        mask = rng.rand(x.shape[0]) < args.visual_dropout
        x[mask, :512] = 0.0
    if modality in ("audio", "joint", "qmf") and args.audio_mute > 0:
        # mute contiguous 30 s segments until the requested fraction is reached
        n = x.shape[0]; seg = 60; k = int(np.ceil(args.audio_mute * n / seg))
        starts = rng.choice(max(1, n - seg), size=min(k, max(1, n // seg)), replace=False)
        off = 512 if modality in ("joint", "qmf") else 0
        for s in starts: x[s:s + seg, off:] = 0.0
    return x

def main(args):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(os.path.join(MODELS, args.tag + ".pt"), map_location="cpu", weights_only=False)
    modality = ck["args"]["modality"]; model = QMFNet(512, ck["in_dim"] - 512) if modality == "qmf" else build_model(ck["in_dim"]); model.load_state_dict(ck["state"]); model.to(dev).eval()
    out_tag = args.out_tag or args.tag
    games = [g for g in games_for(args.split) if has_game(g, need_audio=modality != "visual")]
    if args.league: games = [g for g in games if league_of(g) == args.league]
    rng = np.random.RandomState(1234 + args.seed_offset)
    n = 0
    for g in games:
        try:
            _ = [np.load(visual_path(g, h), mmap_mode="r").shape for h in (1, 2)]
        except Exception as e:
            print("skip unreadable", g, e, flush=True); continue
        for h in (1, 2):
            xv = load_half(g, h, "visual")
            if modality == "visual": x = xv
            elif modality == "audio": x = align(load_half(g, h, "audio"), xv)
            else: x = np.concatenate([xv, align(load_half(g, h, "audio"), xv)], axis=1)
            x = degrade(x, modality, args, rng)
            with torch.no_grad():
                out = model(torch.tensor(x, device=dev).T.unsqueeze(0))
                z = (out[0] if modality == "qmf" else out)[0].T.cpu().numpy()   # qmf: fused logits over 17 events + background
            d = os.path.join(RESULTS, "preds", out_tag, g); os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, f"{h}.npy"), z.astype(np.float16)); n += 1
    print(f"wrote {n} halves for {out_tag} ({args.split})", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True); p.add_argument("--split", default="test"); p.add_argument("--out-tag", default=None)
    p.add_argument("--visual-dropout", type=float, default=0.0); p.add_argument("--audio-mute", type=float, default=0.0)
    p.add_argument("--league", default=None); p.add_argument("--seed-offset", type=int, default=0)
    main(p.parse_args())
