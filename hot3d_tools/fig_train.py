"""Training curves from Lightning metrics.csv files (one or more runs).

  python hot3d_tools/fig_train.py figs/train_<name>.png <label>=<metrics.csv> [<label>=<metrics.csv> ...]
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read(path):
    rows = list(csv.DictReader(open(path)))
    col = lambda k: np.array([(float(r["step"]), float(r[k])) for r in rows if r.get(k) not in (None, "")])
    return {k: col(k) for k in ["fit/pred_loss", "fit/sigreg_loss", "validate/pred_loss_epoch"]}


def smooth(y, n=50):
    if len(y) < n:
        return y
    return np.convolve(y, np.ones(n) / n, "valid")


out, runs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for label, path in runs:
    m = read(path)
    for j, k in enumerate(["fit/pred_loss", "fit/sigreg_loss"]):
        d = m[k]
        if len(d):
            ax[j].plot(d[:, 0][len(d) - len(smooth(d[:, 1])):], smooth(d[:, 1]), label=label, lw=1)
    d = m["validate/pred_loss_epoch"]
    if len(d):
        ax[2].plot(d[:, 0], d[:, 1], "o-", label=label, ms=3)
for a, t in zip(ax, ["train pred loss (smoothed)", "train SIGReg loss (smoothed)", "val pred loss (per epoch)"]):
    a.set_title(t, fontsize=9); a.set_xlabel("step"); a.set_yscale("log"); a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=110)
print("SAVED", out)
