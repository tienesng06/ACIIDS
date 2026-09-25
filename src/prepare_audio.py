"""Extract the public VGGish audio features (Vanderplaetse & Dupont 2020 release) into the artifact layout.
Input : data/audio/vggish_features_download (zip; VGGFeatures/<league>/<season>/<game>/{1,2}_VGGish.npy, (T,512) float64 at 2 fps)
Output: data/audio/features/<league>/<season>/<game>/{1,2}_audio.npy as float16 (same T, 512)
"""
import zipfile, io, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import AUDIO
src = os.path.join(AUDIO, "vggish_features_download"); out = os.path.join(AUDIO, "features")
z = zipfile.ZipFile(src); names = [n for n in z.namelist() if n.endswith(".npy")]
n_done = 0
for n in names:
    rel = n.split("VGGFeatures/")[1]; game, fn = rel.rsplit("/", 1); half = fn[0]
    dst = os.path.join(out, game, f"{half}_audio.npy")
    if os.path.exists(dst): n_done += 1; continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    a = np.load(io.BytesIO(z.read(n)))
    np.save(dst, a.astype(np.float16)); n_done += 1
    if n_done % 100 == 0: print(f"{n_done}/{len(names)}", flush=True)
print("AUDIO PREP DONE", n_done, flush=True)
