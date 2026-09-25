import sys, os
from SoccerNet.Downloader import SoccerNetDownloader, getListGames
split = sys.argv[1]; part = int(sys.argv[2]); nparts = int(sys.argv[3])
d = SoccerNetDownloader(LocalDirectory="data/SoccerNet")
games = getListGames(split)[part::nparts]
for i, g in enumerate(games):
    for f in ["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"]:
        p = os.path.join("data/SoccerNet", g, f)
        if os.path.exists(p) and os.path.getsize(p) > 1e6: continue
        for attempt in range(3):
            try:
                d.downloadGame(game=g, files=[f], spl=split, verbose=False); break
            except Exception as e:
                print("retry", g, f, e, flush=True)
    print(f"{split} part{part}: {i+1}/{len(games)} {g}", flush=True)
print(f"DONE {split} part{part}", flush=True)
