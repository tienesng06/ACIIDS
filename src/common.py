"""Shared utilities for the SN-Agree artifact (SoccerNet-v2 audio-visual agreement study).

Everything is deterministic given a seed. Features are the public SoccerNet-v2 ResNet-152 PCA-512
visual features at 2 fps and the public VGGish audio features released by Vanderplaetse & Dupont (2020).
"""
import os, json, random, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "SoccerNet")
AUDIO = os.path.join(ROOT, "data", "audio")
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
FPS = 2  # features per second

from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Downloader import getListGames
NUM_CLASSES = 17
CLASSES = [INVERSE_EVENT_DICTIONARY_V2[i] for i in range(NUM_CLASSES)]
LEAGUES = ["england_epl", "europe_uefa-champions-league", "france_ligue-1", "germany_bundesliga", "italy_serie-a", "spain_laliga"]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)
    except Exception:
        pass

def games_for(split):
    return getListGames(split)

def league_of(game):
    return game.split("/")[0]

def visual_path(game, half):
    return os.path.join(DATA, game, f"{half}_ResNET_TF2_PCA512.npy")

def audio_path(game, half):
    """Audio features are matched by game folder under data/audio after extraction (see prepare_audio.py)."""
    return os.path.join(AUDIO, "features", game, f"{half}_audio.npy")

def has_game(game, need_audio=True):
    ok = all(os.path.exists(visual_path(game, h)) and os.path.getsize(visual_path(game, h)) > 8e6 for h in (1, 2)) and os.path.exists(os.path.join(DATA, game, "Labels-v2.json"))
    if need_audio:
        ok = ok and all(os.path.exists(audio_path(game, h)) for h in (1, 2))
    return ok

def load_labels(game):
    with open(os.path.join(DATA, game, "Labels-v2.json")) as f:
        return json.load(f)["annotations"]

def label_frames(game, n_frames_half):
    """Return per-half arrays: frame index, class index, visible flag for every annotated event."""
    out = {1: [], 2: []}
    for a in load_labels(game):
        half = int(a["gameTime"][0])
        if half not in (1, 2) or a["label"] not in EVENT_DICTIONARY_V2:
            continue
        pos_ms = int(a["position"]); frame = int(FPS * (pos_ms / 1000.0))   # floor, as in the official label2vector
        frame = min(frame, n_frames_half[half] - 1)
        out[half].append((frame, EVENT_DICTIONARY_V2[a["label"]], 1 if a.get("visibility", "visible") == "visible" else 0))
    return out

def load_half(game, half, modality):
    if modality == "visual":
        x = np.load(visual_path(game, half)).astype(np.float32)
    elif modality == "audio":
        x = np.load(audio_path(game, half)).astype(np.float32)
    else:
        raise ValueError(modality)
    return x

def align(xa, xv):
    """Align audio rows to the visual timeline. Both streams are 2 fps from t=0, so we truncate or zero-pad the
    audio to the visual length (145/778 checked halves differ, almost all by one frame, a few by <40 frames)."""
    n = xv.shape[0]
    if xa.shape[0] == n:
        return xa
    if xa.shape[0] > n:
        return xa[:n]
    return np.concatenate([xa, np.zeros((n - xa.shape[0], xa.shape[1]), dtype=xa.dtype)], axis=0)

def dense_targets(events, n, radius=1):
    """(n, NUM_CLASSES) binary targets: 1 within +-radius frames of an event of that class."""
    y = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    for f, c, _ in events:
        lo, hi = max(0, f - radius), min(n - 1, f + radius)
        y[lo:hi + 1, c] = 1.0
    return y

def nms_1d(scores, window, thresh=0.0):
    """Greedy temporal NMS on a 1-D score array. Returns list of (frame, score)."""
    s = scores.copy(); out = []
    while True:
        i = int(np.argmax(s)); v = float(s[i])
        if v <= thresh: break
        out.append((i, v))
        lo, hi = max(0, i - window), min(len(s), i + window + 1)
        s[lo:hi] = -1.0
    return out

def ensure_dirs():
    for d in (RESULTS, MODELS, os.path.join(RESULTS, "preds")):
        os.makedirs(d, exist_ok=True)
