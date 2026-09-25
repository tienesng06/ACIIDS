"""Re-apply our two changes to the official CALF code (SoccerNet/sn-spotting, CALF/) so it can be run for several seeds.

The published code fixes the seed at 0 inside main.py and writes every run's test predictions to one shared
`outputs/` folder, so parallel runs overwrite each other. Both changes are mechanical and leave the method untouched:
  1. main.py: `torch.manual_seed(0)/np.random.seed(0)` at import time  ->  a `set_seed(s)` helper called from main(),
     plus a `--seed` argument (default 0, i.e. the original behaviour).
  2. train.py: predictions2json(..., "outputs/", ...)  ->  "outputs_<model_name>/", so each run keeps its own folder.
Originals are kept as main.py.orig / train.py.orig.

Usage: python calf_multiseed.py /path/to/CALF
"""
import os, sys

def patch(root):
    m = os.path.join(root, "src", "main.py"); t = os.path.join(root, "src", "train.py")
    s = open(m).read()
    if "def set_seed(" not in s:
        old = "# Fixing seeds for reproducibility\ntorch.manual_seed(0)\nnp.random.seed(0)\n"
        assert s.count(old) == 1, "seed block not found in main.py"
        s = s.replace(old, "# Fixing seeds for reproducibility (settable, so the baseline can be run for several seeds)\n"
                           "def set_seed(s):\n    torch.manual_seed(s); np.random.seed(s); random.seed(s)\n"
                           "    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)\n")
        if "import random" not in s: s = s.replace("import numpy as np", "import numpy as np\nimport random", 1)
        arg = "    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')"
        assert s.count(arg) == 1, "loglevel argument not found"
        s = s.replace(arg, arg + "\n    parser.add_argument('--seed', required=False, type=int, default=0, help='random seed')")
        s = s.replace("def main(args):", "def main(args):\n    set_seed(args.seed)", 1)
        if not os.path.exists(m + ".orig"): open(m + ".orig", "w").write(open(m).read())
        open(m, "w").write(s); print("patched main.py")
    else: print("main.py already patched")
    s = open(t).read()
    if "outputs_" not in s:
        old = '            predictions2json(detections_numpy[index*2], detections_numpy[(index*2)+1],"outputs/", list_game[index], model.framerate)'
        assert s.count(old) == 1, "predictions2json call not found in train.py"
        s = s.replace(old, '            predictions2json(detections_numpy[index*2], detections_numpy[(index*2)+1], os.path.join("outputs_" + str(model_name), ""), list_game[index], model.framerate)')
        if not s.startswith("import os") and "\nimport os" not in s: s = "import os\n" + s
        if not os.path.exists(t + ".orig"): open(t + ".orig", "w").write(open(t).read())
        open(t, "w").write(s); print("patched train.py")
    else: print("train.py already patched")

if __name__ == "__main__":
    patch(sys.argv[1] if len(sys.argv) > 1 else "CALF")
