"""Train a single-representation spotting head (visual-only, audio-only, or early-fused 'joint').

Model: temporal dilated 1-D CNN producing per-frame class probabilities at 2 fps.
Usage: python src/train_branch.py --modality visual --seed 0 [--exclude-league X] [--games-limit N]
"""
import os, sys, argparse, time, json
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from common import *

def build_model(in_dim, hidden=256, dropout=0.2, out_dim=NUM_CLASSES):
    layers = [nn.Conv1d(in_dim, hidden, 1), nn.ReLU(), nn.Dropout(dropout)]
    for d in (1, 2, 4, 8):
        layers += [nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d), nn.ReLU(), nn.Dropout(dropout)]
    layers += [nn.Conv1d(hidden, out_dim, 1)]
    return nn.Sequential(*layers)

class QMFNet(nn.Module):
    """QMF (Zhang et al., ICML 2023), ported from the official code (github.com/QingyangZhang/QMF, models/late_fusion.py and
    train_qmf.py): one classifier per modality, per-sample confidence = 0.1 * logsumexp(logits) (energy score), fused
    logits = sum_m logits_m * conf_m (confidence detached), trained with CE on each modality and on the fusion plus the
    correctness ranking loss (CRL) on each confidence. Here a 'sample' is a frame and the classifier is a softmax over
    the 17 event classes plus background, which is the setting the energy score is defined for."""
    def __init__(self, dv, da, hidden=256, dropout=0.2):
        super().__init__(); self.dv = dv; self.v = build_model(dv, hidden, dropout, NUM_CLASSES + 1); self.a = build_model(da, hidden, dropout, NUM_CLASSES + 1)
    def forward(self, x):
        zv, za = self.v(x[:, :self.dv]), self.a(x[:, self.dv:])                    # (B, C+1, T)
        cv, ca = 0.1 * torch.logsumexp(zv, dim=1), 0.1 * torch.logsumexp(za, dim=1)  # (B, T)
        zf = zv * cv.detach().unsqueeze(1) + za * ca.detach().unsqueeze(1)
        return zf, zv, za, cv, ca

class History:
    """Per-frame cumulative-loss history for CRL, as in the official utils.History (correctness := per-sample loss)."""
    def __init__(self, n): self.corr = np.zeros(n, dtype=np.float64); self.conf = np.zeros(n, dtype=np.float32)
    def update(self, idx, loss, conf): np.add.at(self.corr, idx, loss); self.conf[idx] = conf
    def target_margin(self, i1, i2):
        lo, hi = self.corr.min(), self.corr.max(); den = max(hi - lo, 1e-12)
        c1, c2 = (self.corr[i1] - lo) / den, (self.corr[i2] - lo) / den
        target = (c1 > c2).astype(np.float32) - (c1 < c2).astype(np.float32); margin = np.abs(c1 - c2).astype(np.float32)
        return target, margin

def rank_loss(conf, idx, hist, dev):
    """Official CRL port; equal-history pairs have target 0 and therefore contribute exactly zero loss."""
    idx2 = np.roll(idx, -1); target, margin = hist.target_margin(idx, idx2)
    c1 = conf.reshape(-1, 1); c2 = torch.roll(conf, -1).reshape(-1, 1)
    t = torch.tensor(target, device=dev); tn = t.clone(); tn[tn == 0] = 1
    c2 = c2 + (torch.tensor(margin, device=dev) / tn).reshape(-1, 1)
    return nn.MarginRankingLoss(margin=0.0)(c1, c2, -t.reshape(-1, 1))

class HalfStore:
    """Loads all halves of a split into memory (float16 to save RAM)."""
    def __init__(self, games, modality, radius=1):
        self.x, self.y, self.meta = [], [], []
        for g in games:
            try:
                xv = {h: load_half(g, h, "visual") for h in (1, 2)}
                n = {h: xv[h].shape[0] for h in (1, 2)}
                ev = label_frames(g, n)
                for h in (1, 2):
                    if modality == "visual": x = xv[h]
                    elif modality == "audio": x = align(load_half(g, h, "audio"), xv[h])
                    elif modality in ("joint", "qmf"): x = np.concatenate([xv[h], align(load_half(g, h, "audio"), xv[h])], axis=1)
                    self.x.append(x.astype(np.float16)); self.y.append(dense_targets(ev[h], x.shape[0], radius)); self.meta.append((g, h))
            except Exception as e:
                print("skip", g, e, flush=True)
        self.in_dim = self.x[0].shape[1]; self.dv = 512
        self.offset = np.cumsum([0] + [x.shape[0] for x in self.x])   # global frame index base per half (for the CRL history)

def train(args):
    seed_all(args.seed); ensure_dirs()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_games = [g for g in games_for("train") if has_game(g, need_audio=args.modality != "visual")]
    valid_games = [g for g in games_for("valid") if has_game(g, need_audio=args.modality != "visual")]
    if args.exclude_league:
        train_games = [g for g in train_games if league_of(g) != args.exclude_league]
        valid_games = [g for g in valid_games if league_of(g) != args.exclude_league]
    if args.games_limit: train_games, valid_games = train_games[:args.games_limit], valid_games[:max(2, args.games_limit // 3)]
    print(f"[{args.modality} seed{args.seed}] train games {len(train_games)} valid games {len(valid_games)} device {dev}", flush=True)
    tr, va = HalfStore(train_games, args.modality), HalfStore(valid_games, args.modality)
    qmf = args.modality == "qmf"
    model = (QMFNet(tr.dv, tr.in_dim - tr.dv) if qmf else build_model(tr.in_dim)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    pos_w = torch.full((NUM_CLASSES, 1), args.pos_weight, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    if qmf:   # softmax over 17 events + background; event classes weighted like the BCE positives
        cw = torch.cat([torch.full((NUM_CLASSES,), args.pos_weight), torch.ones(1)]).to(dev)
        ce = nn.CrossEntropyLoss(weight=cw); ce_none = nn.CrossEntropyLoss(weight=cw, reduction="none")
        hist_v, hist_a = History(int(tr.offset[-1])), History(int(tr.offset[-1]))
        def to_index(y):   # multi-hot (B, C, T) -> class index (B, T), background = C
            yi = torch.full(y.shape[::2], NUM_CLASSES, device=dev, dtype=torch.long); m = y.sum(1) > 0
            yi[m] = y.argmax(1)[m]; return yi
        def qmf_loss(out, yb, idx, train_mode):   # idx: global frame index of every (clip, frame) in batch order
            zf, zv, za, cv, ca = out; yi = to_index(yb)
            l = ce(zv, yi) + ce(za, yi) + ce(zf, yi)
            if not train_mode: return l
            lv, la = ce_none(zv, yi).detach().reshape(-1).cpu().numpy(), ce_none(za, yi).detach().reshape(-1).cpu().numpy()
            hist_v.update(idx, lv, cv.detach().reshape(-1).cpu().numpy()); hist_a.update(idx, la, ca.detach().reshape(-1).cpu().numpy())
            return l + rank_loss(cv.reshape(-1), idx, hist_v, dev) + rank_loss(ca.reshape(-1), idx, hist_a, dev)
    L = args.clip_len; rng = np.random.RandomState(args.seed)
    best, best_state, patience = 1e9, None, 0
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        order = rng.permutation(len(tr.x))
        for bi in range(0, len(order), args.batch):
            xb, yb, ib = [], [], []
            for i in order[bi:bi + args.batch]:
                x, y = tr.x[i], tr.y[i]
                for _ in range(args.clips_per_half):
                    s = rng.randint(0, max(1, x.shape[0] - L))
                    xb.append(x[s:s + L]); yb.append(y[s:s + L]); ib.append(np.arange(s, s + L) + tr.offset[i])
            xb = torch.tensor(np.stack(xb).astype(np.float32), device=dev).transpose(1, 2)
            yb = torch.tensor(np.stack(yb), device=dev).transpose(1, 2)
            if args.feat_dropout > 0:
                mask = (torch.rand(xb.shape[0], 1, xb.shape[2], device=dev) > args.feat_dropout).float(); xb = xb * mask
            opt.zero_grad()
            if qmf: loss = qmf_loss(model(xb), yb, np.concatenate(ib), True)
            else: loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        # validation loss on full halves
        model.eval(); vl = 0.0
        with torch.no_grad():
            for x, y in zip(va.x, va.y):
                xt = torch.tensor(x.astype(np.float32), device=dev).T.unsqueeze(0); yt = torch.tensor(y, device=dev).T.unsqueeze(0)
                vl += (qmf_loss(model(xt), yt, None, False) if qmf else lossf(model(xt), yt)).item()
        vl /= max(1, len(va.x))
        print(f"epoch {epoch+1}/{args.epochs} train {tot/max(1,nb):.4f} valid {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vl < best - 1e-4: best, best_state, patience = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= args.patience: break
    model.load_state_dict(best_state)
    tag = f"{args.modality}_s{args.seed}" + (f"_xl-{args.exclude_league}" if args.exclude_league else "")
    torch.save({"state": model.state_dict(), "in_dim": tr.in_dim, "args": vars(args), "best_valid_loss": best}, os.path.join(MODELS, tag + ".pt"))
    print("saved", tag, "best valid loss", round(best, 4), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--modality", default="visual", choices=["visual", "audio", "joint", "qmf"])
    p.add_argument("--seed", type=int, default=0); p.add_argument("--epochs", type=int, default=25); p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--batch", type=int, default=16); p.add_argument("--clips-per-half", type=int, default=4)
    p.add_argument("--clip-len", type=int, default=240); p.add_argument("--pos-weight", type=float, default=5.0); p.add_argument("--feat-dropout", type=float, default=0.0)
    p.add_argument("--exclude-league", default=None); p.add_argument("--games-limit", type=int, default=0)
    train(p.parse_args())
