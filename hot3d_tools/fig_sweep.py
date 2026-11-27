"""DINOv2 predictor sweep figure: python hot3d_tools/fig_sweep.py <sweep_root> <out.png>
(<sweep_root>/<V>/sweep.json from hot3d_dino_sweep.sbatch)"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

root, out = sys.argv[1], sys.argv[2]
VS = ["none", "head", "head_hands"]
res = {v: {x["name"]: x for x in json.load(open(f"{root}/{v}/sweep.json"))} for v in VS}
names = sorted(res["head"], key=lambda n: res["head"][n]["val_table"])
fig, ax = plt.subplots(figsize=(12, 4.5))
w = 0.27
for i, v in enumerate(VS):
    y = [res[v][n]["val_table"] for n in names]
    ax.bar(np.arange(len(names)) + (i - 1) * w, y, w, label=v)
ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
ax.set_ylim(0.29, 0.34); ax.set_ylabel("best val-table loss (MSE, standardized)")
ax.axhline(min(res["head"][n]["val_table"] for n in names), color="k", lw=0.5)
ax.legend(title="motion variant", fontsize=8)
ax.set_title("LeWM predictor on frozen DINOv2 (cls_grid): val sweep, early stopping on val (epochs 12, patience 3)", fontsize=9)
fig.tight_layout(); fig.savefig(out, dpi=110); print("SAVED", out)
