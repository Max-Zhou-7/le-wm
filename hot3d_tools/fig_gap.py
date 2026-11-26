"""Gap-eval figure from gap_eval_<split>.json.
  python hot3d_tools/fig_gap.py <gap_eval.json> <out.png> [title]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

r = json.load(open(sys.argv[1]))
ms = [g * 100 for g in [1, 2, 4, 8]]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for m, v in r["methods"].items():
    get = lambda key, sub=None: [(v[f"g{g}"].get(key, {}) or {}).get(sub, float("nan")) if sub
                                 else v[f"g{g}"].get(key, float("nan")) for g in [1, 2, 4, 8]]
    ax[0].plot(ms, get("wrist_cm", "mean"), "o-", label=m, ms=3)
    if any(x == x for x in get("pix_mae")):
        ax[1].plot(ms, [x * 255 for x in get("pix_mae")], "o-", label=m, ms=3)
    if any(x == x for x in get("emb_err")):
        ax[2].plot(ms, get("emb_err"), "o-", label=m, ms=3)
for a, t in zip(ax, ["wrist error (cm, mean)", "pixel MAE (/255, vs real frame)", "embedding error (relative)"]):
    a.set_title(t, fontsize=9); a.set_xlabel("gap (ms)"); a.set_xticks(ms); a.legend(fontsize=7)
fig.suptitle(sys.argv[3] if len(sys.argv) > 3 else r["emb_dir"], fontsize=9)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=110)
print("SAVED", sys.argv[2])
