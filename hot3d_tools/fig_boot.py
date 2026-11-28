"""Figure with bootstrap 95% CIs: python hot3d_tools/fig_boot.py <results_v3_dir> <out.png> [--split val]
Row 1: wrist error (last gap frame) with CI bands, DINOv2 head / head_hands and LeWM head.
Row 2: paired difference vs repeat-last; interpolation ablation (all gap frames)."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

G = [1, 2, 4, 8, 16, 32]
ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--split", default="val")
a = ap.parse_args()
ms = [100 * g for g in G]
load = lambda m: json.load(open(f"{a.root}/{m}/boot_{a.split}.json"))


def band(ax, B, sec, m, met, which="stat", label=None, **kw):
    s = [(B.get(f"{sec}|{m}|g{g}|{met}") or {}).get(which) for g in G]
    if any(x is None for x in s):
        return
    y = [x["mean"] for x in s]
    ln, = ax.plot(ms, y, "o-", ms=3, label=label or m, **kw)
    ax.fill_between(ms, [x["lo"] for x in s], [x["hi"] for x in s], alpha=0.18, color=ln.get_color())


fig, ax = plt.subplots(2, 3, figsize=(19, 9))
panels = [("dino_best_head", "M_lewm_head"), ("dino_best_head_hands", "M_lewm_head_hands"), ("lewm_head", "M_lewm_head")]
for c, (model, lm) in enumerate(panels):
    B = load(model)
    for m in ["oracle", "B0_repeat", "B2_timewarp_fill", "kalman_probe", "kalman_damped_probe", "kalman_damped_gt", lm]:
        band(ax[0, c], B, "methods", m, "wrist")
    ax[0, c].set_title(f"{model}: wrist error (cm), last frame of the gap, 95% CI (recording bootstrap)", fontsize=8)
    ax[0, c].set_ylim(0, 40)
for model, lm in panels[:2]:
    B = load(model)
    for m in ["B2_timewarp_fill", "kalman_damped_probe", lm]:
        band(ax[1, 0], B, "methods", m, "wrist", "diff_vs_repeat", label=f"{model}: {m}")
ax[1, 0].axhline(0, color="k", lw=0.6); ax[1, 0].set_ylim(-6, 6)
ax[1, 0].set_title("paired difference vs repeat-last (cm; < 0 = better), 95% CI", fontsize=8)
for c, model in [(1, "dino_best_head_hands"), (2, "lewm_head")]:
    B = load(model)
    for m in ["A_oracle", "A_repeat", "A_lewm", "I_nearest", "I_linear", "I_lewm_endfix"]:
        band(ax[1, c], B, "gapall", m, "wrist")
    ax[1, c].set_title(f"{model}: interpolation ablation, wrist error over ALL dropped frames (cm)", fontsize=8)
for A in ax.flat:
    A.set_xscale("log"); A.set_xticks(ms); A.set_xticklabels([str(x) for x in ms], fontsize=7)
    A.set_xlabel("gap (ms)"); A.legend(fontsize=6)
fig.suptitle(f"Gap eval with recording-level bootstrap CIs ({a.split})", fontsize=10)
fig.tight_layout(); fig.savefig(a.out, dpi=110); print("SAVED", a.out)

# Segment split figure: rows = segments, cols = models; wrist error with 95% CI bands.
segs = ["interaction", "head_only", "other"]
if any(f"seg:{s}|n" in load(panels[0][0]) for s in segs):
    fig, ax = plt.subplots(3, 3, figsize=(19, 13), sharey="row")
    for c, (model, lm) in enumerate(panels):
        B = load(model)
        for r_, s in enumerate(segs):
            for m in ["oracle", "B0_repeat", "B2_timewarp_fill", "kalman_damped_probe", "kalman_damped_gt", lm]:
                band(ax[r_, c], B, f"seg:{s}|methods", m, "wrist")
            n = B.get(f"seg:{s}|n", {})
            ax[r_, c].set_title(f"{model}, segment {s}: wrist error (cm), 95% CI; windows "
                                + ",".join(str(n.get(f'g{g}', 0)) for g in G), fontsize=8)
            ax[r_, c].set_ylim(0, 35)
    for A in ax.flat:
        A.set_xscale("log"); A.set_xticks(ms); A.set_xticklabels([str(x) for x in ms], fontsize=7)
        A.set_xlabel("gap (ms)"); A.legend(fontsize=6)
    fig.suptitle(f"Gap eval by segment ({a.split}); recording-level bootstrap CIs", fontsize=10)
    fig.tight_layout(); o2 = a.out.replace(".png", "_segments.png"); fig.savefig(o2, dpi=100); print("SAVED", o2)
