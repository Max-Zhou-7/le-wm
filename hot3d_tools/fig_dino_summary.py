"""Summary figure: LeWM predictor on frozen DINOv2 (3 action variants) vs repeat-last and TimeWarp-fill, val.
python hot3d_tools/fig_dino_summary.py <dino_summary.json> <out.png>"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = json.load(open(sys.argv[1]))
g = [100, 200, 400, 800]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
base = d["dino_head"]
lines = [("repeat last frame", base["B0_repeat"], "gray", "--"), ("TimeWarp-fill", base["B2_timewarp_fill"], "tab:green", "--"),
         ("probe on the real frame (floor)", base["oracle"], "k", ":")]
for v, c in [("none", "tab:orange"), ("head", "tab:blue"), ("head_hands", "tab:red")]:
    lines.append((f"LeWM-DINOv2, actions: {v}" + (" (hand motion = oracle)" if v == "head_hands" else ""), d[f"dino_{v}"][f"M_lewm_{v}"], c, "-"))
for j, (k, yl, t) in enumerate([("wrist_cm", "wrist error (cm), probe on the forecast", "Wrist position (DINOv2 probe works: oracle 10.5 cm vs average 19.5)"),
                                ("up_deg", "head tilt error (deg)", "Head tilt (world up in camera)"),
                                ("emb_err", "||f_hat - f|| / mean ||f - mean f||", "DINOv2 feature error (LeWM's own objective)")]):
    for name, r, c, ls in lines:
        if k == "emb_err" and name.startswith("probe"):
            continue
        ax[j].plot(g, r[k], marker="o", color=c, ls=ls, label=name, ms=4)
    ax[j].set_xlabel("gap (ms)"); ax[j].set_ylabel(yl); ax[j].set_title(t, fontsize=9); ax[j].set_xticks(g)
ax[0].legend(fontsize=7)
fig.suptitle("LeWM predictor trained on FROZEN DINOv2-S features (CLS + 2x2 grid), val only. "
             "Probes: ridge on train, lambda on val, scored on val.", fontsize=9)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=110)
print("SAVED", sys.argv[2])
