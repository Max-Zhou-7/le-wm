"""Training curves of train_feat.py runs: python hot3d_tools/fig_train_feat.py out.png label=<dir>/train_hist.json ..."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out, runs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for label, path in runs:
    h = json.load(open(path))["hist"]
    ep = [x["epoch"] for x in h]
    for j, k in enumerate(["train", "holdout", "val_table"]):
        ax[j].plot(ep, [x[k] for x in h], "o-", ms=3, label=label)
for j, t in enumerate(["train loss", "held-out clips (10% of train recordings)", "val table (other recordings)"]):
    ax[j].set_title(t, fontsize=9); ax[j].set_xlabel("epoch"); ax[j].set_ylabel("MSE (standardized DINOv2 features)")
    ax[j].set_yscale("log"); ax[j].legend(fontsize=8)
fig.suptitle("LeWM predictor on frozen DINOv2-S features (CLS + 2x2 grid)", fontsize=9)
fig.tight_layout()
fig.savefig(out, dpi=110)
print("SAVED", out)
