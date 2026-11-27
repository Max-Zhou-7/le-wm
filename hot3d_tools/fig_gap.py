"""Gap-eval figure from gap_eval_<split>.json.
  python hot3d_tools/fig_gap.py <gap_eval.json> <out.png> [title]
Panels: wrist error (total), wrist error above the oracle floor (probe on the real frame),
pixel MAE (TimeWarp / repeat), feature error (favours LeWM: its own training objective).
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

r = json.load(open(sys.argv[1]))
ms = r["gaps_ms"]
G = [m // 100 for m in ms]
fig, ax = plt.subplots(1, 4, figsize=(21, 4.3))
for m, v in r["methods"].items():
    if not any(f"g{g}" in v for g in G):
        continue
    get = lambda key, sub=None: [((v.get(f"g{g}", {}).get(key, {}) or {}).get(sub, float("nan")) if sub
                                  else v.get(f"g{g}", {}).get(key, float("nan"))) for g in G]
    ls = "--" if ("gt" in m and "kalman" in m or m == "hold_gt") else "-"
    ax[0].plot(ms, get("wrist_cm", "mean"), "o" + ls, label=m, ms=3)
    if m not in ("oracle",):
        ax[1].plot(ms, get("wrist_above_floor"), "o" + ls, label=m, ms=3)
    if any(x == x for x in get("pix_mae")):
        ax[2].plot(ms, [x * 255 for x in get("pix_mae")], "o-", label=m, ms=3)
    if any(x == x for x in get("emb_err")):
        ax[3].plot(ms, get("emb_err"), "o-", label=m, ms=3)
ax[1].axhline(0, color="k", lw=0.6)
for a, t in zip(ax, ["wrist error (cm, mean, total)", "wrist error above the oracle floor (cm)",
                     "pixel MAE (/255, vs real frame)", "feature error (relative; LeWM's own objective, favours LeWM)"]):
    a.set_title(t, fontsize=9); a.set_xlabel("gap (ms)"); a.set_xscale("log"); a.set_xticks(ms)
    a.set_xticklabels([str(x) for x in ms], fontsize=7); a.legend(fontsize=6)
fig.suptitle((sys.argv[3] if len(sys.argv) > 3 else r["emb_dir"]) + f"  ({r['n_windows']} windows)", fontsize=9)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=110)
print("SAVED", sys.argv[2])
