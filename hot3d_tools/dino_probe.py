"""DINOv2 go/no-go: leave-one-participant-out (LOPO) linear wrist probes on frozen DINOv2 features of
the 7 GT training participants (train + val tables, every 10th row). Test is never loaded.

Versions: cls (384) and cls_grid (CLS + 2x2 averaged patch grid, 1920). Targets: left/right wrist 3D
(cm, camera frame) and 2D (px, 224 image); clean frames = wrist valid & head valid & qa_pass & wrist
inside the image and lens limit (probe_fix.load). Fold p: inner participant q (next in the list) picks
lambda, refit on the other 6, score p. Baseline = mean of the 6 training participants.

Improvement of a fold = 1 - probe_err / mean_err.
GO rule (user, 2026-11-27): the best version beats the average by >= 15% on average AND in >= 5 of 7
folds. Here "on average" = mean over the 4 targets of (1 - sum-over-folds probe / sum mean, i.e. the
fold-averaged errors), and a fold counts if its improvement averaged over the 4 targets is > 0
(the count of folds with >= 15% is reported as well).

  python hot3d_tools/dino_probe.py <dino_emb_dir> <lance_dir> [--fig figs/x.png]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probes import add_bias  # noqa: E402
from probe_fix import load, participants, fit_ridge, err  # noqa: E402

LAMS = [10.0 ** p for p in range(-6, 5)]
VERSIONS = {"cls": slice(0, 384), "cls_grid": slice(0, 1920)}
TARGETS = [("L_3d", 0, "W", "cm"), ("R_3d", 1, "W", "cm"), ("L_2d", 0, "uv", "px"), ("R_2d", 1, "uv", "px")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emb_dir")
    ap.add_argument("lance_dir")
    ap.add_argument("--fig", default="")
    a = ap.parse_args()
    S = {sp: load(a.emb_dir, a.lance_dir, sp) for sp in ["train", "val"]}
    for sp, s in S.items():
        s["P"] = participants(a.lance_dir, sp, s["rows"])
    A = {k: np.concatenate([S["train"][k], S["val"][k]]) for k in ["X", "W", "uv", "clean", "P"]}
    plist = sorted(set(A["P"]))
    rep = dict(emb_dir=a.emb_dir, participants=plist, n_frames=int(len(A["X"])), versions={})
    for vname, sl in VERSIONS.items():
        X = A["X"][:, sl]
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xs = add_bias((X - mu) / sd)
        V = dict(targets={})
        for tname, k, key, unit in TARGETS:
            Y, M = A[key][:, k], A["clean"][:, k]
            fo = {}
            for i, p in enumerate(plist):
                q = plist[(i + 1) % len(plist)]
                te, inn = M & (A["P"] == p), M & (A["P"] == q)
                tr, trq = M & (A["P"] != p) & (A["P"] != q), M & (A["P"] != p)
                lam, _ = fit_ridge(Xs[tr], Y[tr], Xs[inn], Y[inn], LAMS)
                _, f = fit_ridge(Xs[trq], Y[trq], Xs[inn], Y[inn], [lam])
                pe = float(err(f(Xs[te]), Y[te]).mean())
                me = float(err(Y[trq].mean(0), Y[te]).mean())
                fo[p] = dict(n=int(te.sum()), probe=pe, guess_mean=me, impr=1 - pe / me, lam=lam)
            avg_p = float(np.mean([f["probe"] for f in fo.values()]))
            avg_m = float(np.mean([f["guess_mean"] for f in fo.values()]))
            V["targets"][tname] = dict(unit=unit, folds=fo, probe=avg_p, guess_mean=avg_m, impr=1 - avg_p / avg_m,
                                       folds_better=int(sum(f["impr"] > 0 for f in fo.values())))
            print(f"{vname:8s} {tname}: probe {avg_p:6.2f} vs mean {avg_m:6.2f} {unit}  improvement "
                  f"{100 * (1 - avg_p / avg_m):5.1f}%  folds better {V['targets'][tname]['folds_better']}/7  "
                  f"per fold " + " ".join(f"{p}:{100 * f['impr']:+.0f}%" for p, f in fo.items()), flush=True)
        fold_impr = {p: float(np.mean([V["targets"][t[0]]["folds"][p]["impr"] for t in TARGETS])) for p in plist}
        V["avg_impr"] = float(np.mean([V["targets"][t[0]]["impr"] for t in TARGETS]))
        V["avg_impr_3d"] = float(np.mean([V["targets"][t]["impr"] for t in ["L_3d", "R_3d"]]))
        V["fold_impr"] = fold_impr
        V["folds_better"] = int(sum(v > 0 for v in fold_impr.values()))
        V["folds_ge15"] = int(sum(v >= 0.15 for v in fold_impr.values()))
        rep["versions"][vname] = V
        print(f"{vname:8s} ALL: avg improvement {100 * V['avg_impr']:.1f}% (3D only {100 * V['avg_impr_3d']:.1f}%), "
              f"folds better {V['folds_better']}/7, folds >= 15%: {V['folds_ge15']}/7", flush=True)
    best = max(rep["versions"], key=lambda v: rep["versions"][v]["avg_impr"])
    B = rep["versions"][best]
    rep["best"] = best
    rep["decision"] = "GO" if (B["avg_impr"] >= 0.15 and B["folds_better"] >= 5) else "NO-GO"
    print(f"DECISION {rep['decision']} (best version {best}: {100 * B['avg_impr']:.1f}%, folds better {B['folds_better']}/7)")
    json.dump(rep, open(f"{a.emb_dir}/dino_go.json", "w"), indent=1)
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(18, 4.3))
        for j, (tname, _, _, unit) in enumerate(TARGETS):
            xs = np.arange(len(plist))
            ax[j].bar(xs - 0.27, [rep["versions"]["cls"]["targets"][tname]["folds"][p]["guess_mean"] for p in plist], 0.27,
                      color="gray", label="guess the average")
            for i, v in enumerate(VERSIONS):
                ax[j].bar(xs + i * 0.27, [rep["versions"][v]["targets"][tname]["folds"][p]["probe"] for p in plist], 0.27,
                          label=f"{v} ({100 * rep['versions'][v]['targets'][tname]['impr']:+.0f}%)")
            ax[j].set_xticks(xs, plist, fontsize=7)
            ax[j].set_ylabel(f"error ({unit})")
            ax[j].set_title(f"{tname} wrist, held-out participant", fontsize=9)
            ax[j].legend(fontsize=7)
        fig.suptitle(f"DINOv2-S linear wrist probes, leave-one-participant-out (7 training participants, every 10th frame). "
                     f"Decision: {rep['decision']} (best {best}, avg improvement {100 * B['avg_impr']:.1f}%, "
                     f"folds better {B['folds_better']}/7; rule: >= 15% and >= 5/7)", fontsize=9)
        fig.tight_layout()
        fig.savefig(a.fig, dpi=110)
        print("SAVED", a.fig)
    print("DINO PROBE OK")


if __name__ == "__main__":
    main()
