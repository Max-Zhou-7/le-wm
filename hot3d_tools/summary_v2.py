"""Val (or test) summary v2: markdown tables + one figure from ~/hot3d_results/v2/<NAME>/.
  python hot3d_tools/summary_v2.py <results_v2_dir> <out.md> <out.png> [--split val]
"""
import argparse
import glob
import json
import os

import numpy as np

G = [1, 2, 4, 8, 16, 32]
MODELS = ["lewm_none", "lewm_head", "lewm_head_hands", "dino_best_none", "dino_best_head", "dino_best_head_hands"]


def cell(x, f="{:.1f}"):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("md"); ap.add_argument("png")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    S = a.split
    L = []
    data = {}
    for m in MODELS:
        f = f"{a.root}/{m}/gap_eval_{S}_k32.json"
        if os.path.exists(f):
            data[m] = json.load(open(f))
    for m, r in data.items():
        M = r["methods"]
        lm = [k for k in M if k.startswith("M_lewm") and "noise" not in k][0]
        rows = [("oracle (probe on real frame)", "oracle"), ("repeat last", "B0_repeat"), ("TimeWarp-fill", "B2_timewarp_fill"),
                ("Kalman, probe obs.", "kalman_probe"), ("Kalman, GT wrist (best case)", "kalman_gt"),
                ("LeWM forecast", lm)] + [(f"LeWM, wrist noise {k.split('noise')[1]} cm", k) for k in M if "noise" in k]
        L.append(f"\n### {m} ({r['n_windows']} {S} windows)\n")
        L.append("Wrist error, cm: total (above the oracle floor)\n")
        L.append("| method | " + " | ".join(f"{100 * g} ms" for g in G) + " |")
        L.append("|---|" + "---|" * len(G))
        for name, k in rows:
            if k not in M:
                continue
            c = []
            for g in G:
                x = M[k].get(f"g{g}", {})
                tot = x.get("wrist_cm", {}).get("mean")
                fl = x.get("wrist_above_floor")
                c.append(cell(tot) if k == "oracle" else f"{cell(tot)} ({cell(fl, '{:+.1f}')})")
            L.append(f"| {name} | " + " | ".join(c) + " |")
        L.append("\nHead tilt (deg) / feature error (LeWM's own objective, favours LeWM) / pixel MAE (/255)\n")
        L.append("| method | " + " | ".join(f"{100 * g} ms" for g in G) + " |")
        L.append("|---|" + "---|" * len(G))
        for name, k in rows[:3] + [rows[5]]:
            if k not in M:
                continue
            c = []
            for g in G:
                x = M[k].get(f"g{g}", {})
                pm = x.get("pix_mae")
                c.append(f"{cell(x.get('up_deg', {}).get('mean'))} / {cell(x.get('emb_err'), '{:.2f}')} / "
                         f"{cell(pm * 255 if pm is not None else None)}")
            L.append(f"| {name} | " + " | ".join(c) + " |")
        kp = r.get("kalman_params", {})
        L.append("\nKalman (q, r) per gap: " + "; ".join(f"{t}: " + ", ".join(f"{g}:{tuple(v)}" for g, v in d.items())
                                                  for t, d in kp.items()))
    # RQ2
    L.append(f"\n## RQ2 ({S})\n")
    for m in MODELS:
        f = f"{a.root}/{m}/rq2_{S}_k32.json"
        if not os.path.exists(f):
            continue
        q = json.load(open(f))
        p1 = q["part1"]
        L.append(f"\n### {m}\n")
        L.append("Surprise vs gap (mean; repeat-last in brackets): " + ", ".join(
            f"{100 * k} ms {p1['surprise_mean'][k - 1]:.2f} ({p1['repeat_mean'][k - 1]:.2f})" for k in G))
        L.append("\n| budget | kind | kept | feat err, LeWM fill | feat err, last-kept fill | pixel MAE (/255) |")
        L.append("|---|---|---|---|---|---|")
        nm = {"fixed": "fixed rate", "surprise": "surprise ORACLE", "lastsurp": "surprise realistic", "head": "head motion"}
        for b in ["1/2", "1/4", "1/8"]:
            for kd in ["fixed", "surprise", "lastsurp", "head"]:
                for x in q["part2"]["results"]:
                    if x["budget"] == b and x["kind"] == kd:
                        L.append(f"| {b} | {nm[kd]} | {x['kept_frac']:.3f} | {x['feat_err_lewm']:.3f} | "
                                 f"{x['feat_err_repeat']:.3f} | {255 * x['pix_mae_repeat']:.1f} |")
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    # figure: wrist total and above floor for the best model per encoder
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = [100 * g for g in G]
    fig, ax = plt.subplots(2, 3, figsize=(18, 8))
    for row, enc in enumerate(["lewm", "dino_best"]):
        for col, (key, ttl) in enumerate([("wrist_cm", "wrist error total (cm)"), ("wrist_above_floor", "wrist above oracle floor (cm)")]):
            A = ax[row, col]
            for v in ["none", "head", "head_hands"]:
                r = data.get(f"{enc}_{v}")
                if not r:
                    continue
                M = r["methods"]
                for k in M:
                    if not k.startswith("M_lewm"):
                        continue
                    y = [(M[k][f"g{g}"].get(key, {}) or {}).get("mean") if key == "wrist_cm" else M[k][f"g{g}"].get(key) for g in G]
                    A.plot(ms, y, "o-" if "noise" not in k else "x:", ms=3, label=k)
                if v == "head":
                    for k, s in [("B0_repeat", "s--"), ("B2_timewarp_fill", "^--"), ("kalman_probe", "d--"), ("kalman_gt", "v:")]:
                        y = [(M[k][f"g{g}"].get(key, {}) or {}).get("mean") if key == "wrist_cm" else M[k][f"g{g}"].get(key) for g in G]
                        A.plot(ms, y, s, ms=3, color="gray" if k != "kalman_gt" else "k", alpha=0.8, label=k)
                    if key == "wrist_cm":
                        A.plot(ms, [M["oracle"][f"g{g}"]["wrist_cm"]["mean"] for g in G], "k-", lw=0.6, label="oracle floor")
            A.set_xscale("log"); A.set_xticks(ms); A.set_xticklabels([str(x) for x in ms], fontsize=7)
            A.set_title(f"{enc}: {ttl}", fontsize=9); A.set_xlabel("gap (ms)"); A.legend(fontsize=6)
        A = ax[row, 2]
        r = data.get(f"{enc}_head")
        if r:
            M = r["methods"]
            for k in ["B0_repeat", "B2_timewarp_fill"] + [k for k in M if k.startswith("M_lewm")]:
                y = [M[k][f"g{g}"].get("emb_err") for g in G]
                A.plot(ms, y, "o-", ms=3, label=k)
            for v in ["none", "head_hands"]:
                rv = data.get(f"{enc}_{v}")
                if rv:
                    k = [k for k in rv["methods"] if k.startswith("M_lewm") and "noise" not in k][0]
                    A.plot(ms, [rv["methods"][k][f"g{g}"].get("emb_err") for g in G], "o-", ms=3, label=k)
        A.set_xscale("log"); A.set_xticks(ms); A.set_xticklabels([str(x) for x in ms], fontsize=7)
        A.set_title(f"{enc}: feature error (own objective, favours LeWM)", fontsize=9); A.legend(fontsize=6)
    fig.suptitle(f"Gap forecasting, {S}: LeWM (top) vs LeWM predictor on frozen DINOv2 (bottom)", fontsize=10)
    fig.tight_layout(); fig.savefig(a.png, dpi=110)
    print("SAVED", a.png, a.md)


if __name__ == "__main__":
    main()
