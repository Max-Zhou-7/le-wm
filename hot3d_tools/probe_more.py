"""More probes on z (192-d encoder+projector embedding), leave-one-participant-out (LOPO) over the 7
GT training participants (train table stride 3 + val table subsampled to stride 3). Test is scored
for REPORTING ONLY (fit on all 7, hyperparameter = most common fold choice).

Per fold p: inner participant q (next in the list) picks lambda / C; refit on all 6 others.
All frames need head_valid & qa_pass. Probes are linear (ridge / logistic): the MLP did not help
for wrists across people (probe_fix.py lopo).

  1. objects: per object ID, position in the camera frame (cm, 3D) and in the image (px, 2D; only
     frames where the object origin projects inside the image and lens limit). Valid = obj_valid.
     An object is probed if >= 3 of the 7 participants have >= 50 frames of it; folds only over those.
     Baseline: training-people mean.
  2. yes/no (logistic, standardized z), baseline = most common answer among training people:
     inview_L/R   wrist inside the 224 pinhole image and lens limit (frames: that hand valid)
     fast_L/R     wrist world speed > 30 cm/s, from rows r -> r+1 (both valid, same recording, dt <= 50 ms)
     near         some valid wrist within 5 cm of some valid object ORIGIN (frames: >= 1 hand and >= 1
                  object valid). Note: the table has no fingertip/mesh positions, so this is wrist to
                  object origin, which is rarely < 5 cm; prevalence is reported.
     Metrics: accuracy (threshold 0.5), balanced accuracy, AUC (majority: bal.acc 0.5, AUC 0.5).
  3. head tilt: world up (+Z) in the camera frame, angle error (deg); baseline = mean direction.

  python hot3d_tools/probe_more.py <emb_dir> <lance_dir> [--fig figs/x.png]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probes import targets, add_bias, ang_err  # noqa: E402
from probe_fix import fit_ridge, err  # noqa: E402

IMG, LENS = 224, 1.0
LAMS = [10.0 ** p for p in range(-6, 5)]
CS = [1e-3, 1e-2, 1e-1, 1.0]
MIN_FR = 50


def project(P, K):
    """P [...,3] cm camera frame -> uv [...,2], inside [...]"""
    z = P[..., 2]
    zs = np.where(z > 1e-6, z, 1.0)
    uv = np.stack([K[0, 0] * P[..., 0] / zs + K[0, 2], K[1, 1] * P[..., 1] / zs + K[1, 2]], -1)
    ang = np.arccos(np.clip(z / np.linalg.norm(P, axis=-1).clip(1e-9), -1, 1))
    ins = (z > 1e-6) & (uv[..., 0] >= 0) & (uv[..., 0] < IMG) & (uv[..., 1] >= 0) & (uv[..., 1] < IMG) & (ang < LENS)
    return uv, ins


def load(emb_dir, lance_dir, split, K):
    import lance
    X = np.load(f"{emb_dir}/emb_{split}.npy").astype(np.float64)
    rows = np.load(f"{emb_dir}/rows_{split}.npy")
    T = targets(lance_dir, split, 110, rows, "z", qa=False)
    ds = lance.dataset(f"{lance_dir}/hot3d_{split}_fov110.lance")
    n_tab = ds.count_rows()
    nxt = np.minimum(rows + 1, n_tab - 1)
    cols = ["qa_pass", "head_valid", "hand_valid", "hand_wrist", "t_sec", "recording"]
    a = ds.take(rows.tolist(), columns=cols)
    b = ds.take(nxt.tolist(), columns=cols)
    g = lambda tab, c: np.asarray(tab[c].to_pylist(), np.float64)
    N = len(rows)
    rec = np.asarray(a["recording"].to_pylist())
    hv = g(a, "head_valid").reshape(N) > 0.5
    base = hv & (g(a, "qa_pass").reshape(N) > 0.5)
    hav = g(a, "hand_valid").reshape(N, 2) > 0.5
    # speed (world), r -> r+1
    w0 = g(a, "hand_wrist").reshape(N, 2, 7)[..., :3]
    w1 = g(b, "hand_wrist").reshape(N, 2, 7)[..., :3]
    dt = g(b, "t_sec").reshape(N) - g(a, "t_sec").reshape(N)
    sp_ok = hav & (g(b, "hand_valid").reshape(N, 2) > 0.5) & (rec == np.asarray(b["recording"].to_pylist()))[:, None] \
        & ((dt > 1e-3) & (dt <= 0.05) & (nxt == rows + 1))[:, None]
    speed = np.linalg.norm(w1 - w0, axis=-1) * 100.0 / np.where(dt > 1e-3, dt, 1.0)[:, None]   # cm/s
    W = T["wrist"].reshape(N, 2, 3)
    _, w_in = project(W, K)
    obj = T["obj"]                                     # [N,O,3] cm camera
    ov = T["obj_ok"]                                   # obj_valid & head_valid
    o_uv, o_in = project(obj, K)
    d = np.linalg.norm(W[:, :, None] - obj[:, None], axis=-1)           # [N,2,O]
    pair = hav[:, :, None] & ov[:, None, :]
    near = (np.where(pair, d, np.inf) < 5.0).any((1, 2))
    near_ok = base & hav.any(1) & ov.any(1)
    P = np.array([r.split("_")[0] for r in rec])
    return dict(X=X, P=P, base=base, hav=hav, w_in=w_in, speed=speed, sp_ok=sp_ok, obj=obj, ov=ov,
                o_uv=o_uv, o_in=o_in, near=near, near_ok=near_ok, up=T["up"], rows=rows)


def logistic(Xtr, ytr, Xin, yin, Cs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    best = None
    for C in Cs:
        if len(np.unique(ytr)) < 2:
            return C, None
        m = LogisticRegression(C=C, max_iter=300).fit(Xtr, ytr)
        s = roc_auc_score(yin, m.decision_function(Xin)) if len(np.unique(yin)) == 2 else 0.5
        if best is None or s > best[0]:
            best = (s, C, m)
    return best[1], best[2]


def cls_metrics(y, score, pred):
    from sklearn.metrics import roc_auc_score
    acc = float((pred == y).mean())
    pos, neg = y == 1, y == 0
    bal = float(0.5 * ((pred[pos] == 1).mean() if pos.any() else np.nan) + 0.5 * ((pred[neg] == 0).mean() if neg.any() else np.nan))
    auc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")
    return dict(n=int(len(y)), pos_rate=float(y.mean()), acc=acc, bal_acc=bal, auc=auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emb_dir")
    ap.add_argument("lance_dir")
    ap.add_argument("--fig", default="")
    a = ap.parse_args()
    K = np.asarray(json.load(open(f"{a.lance_dir}/hot3d_fov110_meta.json"))["K"])
    S = {sp: load(a.emb_dir, a.lance_dir, sp, K) for sp in ["train", "val", "test"]}
    v = S["val"]
    keep = np.flatnonzero(v["rows"] % 3 == 0)
    A = {k: np.concatenate([S["train"][k], v[k][keep]]) for k in S["train"]}
    T = S["test"]
    mu, sd = A["X"].mean(0), A["X"].std(0) + 1e-6
    for s in (A, T):
        s["Xs"] = add_bias((s["X"] - mu) / sd)
        s["Xn"] = (s["X"] - mu) / sd
    plist = sorted(set(A["P"]))
    rep = dict(emb_dir=a.emb_dir, participants=plist, note="test = report only", reg={}, cls={})

    def lopo_reg(name, Y, M, ferr, unit, folds_over=None):
        fo, lams_ = {}, []
        fl = folds_over or plist
        for i, p in enumerate(fl):
            q = fl[(i + 1) % len(fl)]
            te, inn = M & (A["P"] == p), M & (A["P"] == q)
            tr, trq = M & (A["P"] != p) & (A["P"] != q), M & (A["P"] != p)
            if te.sum() < MIN_FR or inn.sum() < 10 or tr.sum() < 100:
                continue
            lam, _ = fit_ridge(A["Xs"][tr], Y[tr], A["Xs"][inn], Y[inn], LAMS)
            _, f = fit_ridge(A["Xs"][trq], Y[trq], A["Xs"][inn], Y[inn], [lam])
            mean = Y[trq].mean(0)
            fo[p] = dict(n=int(te.sum()), probe=float(ferr(f(A["Xs"][te]), Y[te]).mean()),
                         guess_mean=float(ferr(np.broadcast_to(mean, Y[te].shape), Y[te]).mean()), lam=lam)
            lams_.append(lam)
        return fo, lams_

    def test_reg(Y, M, Yt, Mt, ferr, lam):
        if Mt.sum() < MIN_FR:
            return None
        _, f = fit_ridge(A["Xs"][M], Y[M], A["Xs"][M][:10], Y[M][:10], [lam])
        mean = Y[M].mean(0)
        return dict(n=int(Mt.sum()), probe=float(ferr(f(T["Xs"][Mt]), Yt[Mt]).mean()),
                    guess_mean=float(ferr(np.broadcast_to(mean, Yt[Mt].shape), Yt[Mt]).mean()))

    # 1. objects
    O = A["obj"].shape[1]
    for tgt, unit in [("3d", "cm"), ("2d", "px")]:
        per_obj = {}
        for o in range(O):
            M = A["base"] & A["ov"][:, o]
            Mt = T["base"] & T["ov"][:, o]
            if tgt == "2d":
                M, Mt = M & A["o_in"][:, o], Mt & T["o_in"][:, o]
                Y, Yt = A["o_uv"][:, o], T["o_uv"][:, o]
            else:
                Y, Yt = A["obj"][:, o], T["obj"][:, o]
            have = [p for p in plist if (M & (A["P"] == p)).sum() >= MIN_FR]
            if len(have) < 3:
                continue
            fo, lams_ = lopo_reg(f"obj{o}_{tgt}", Y, M, err, unit, have)
            if not fo:
                continue
            lam = max(set(lams_), key=lams_.count)
            per_obj[o] = dict(folds=fo, test=test_reg(Y, M, Yt, Mt, err, lam))
        fl = [f for d in per_obj.values() for f in d["folds"].values()]
        tl = [d["test"] for d in per_obj.values() if d["test"]]
        wavg = lambda L, k: float(sum(x[k] * x["n"] for x in L) / max(sum(x["n"] for x in L), 1))
        rep["reg"][f"objects_{tgt}"] = dict(
            unit=unit, n_objects=len(per_obj),
            lopo=dict(probe=wavg(fl, "probe"), guess_mean=wavg(fl, "guess_mean"), n=sum(x["n"] for x in fl),
                      objects_probe_better=int(sum(np.mean([f["probe"] for f in d["folds"].values()]) <
                                                   np.mean([f["guess_mean"] for f in d["folds"].values()]) for d in per_obj.values()))),
            test=dict(probe=wavg(tl, "probe"), guess_mean=wavg(tl, "guess_mean"), n=sum(x["n"] for x in tl), n_objects=len(tl)),
            per_object={int(o): d for o, d in per_obj.items()})
        r = rep["reg"][f"objects_{tgt}"]
        print(f"objects {tgt}: {r['n_objects']} objects | LOPO probe {r['lopo']['probe']:.2f} vs mean {r['lopo']['guess_mean']:.2f} {unit} "
              f"(probe better for {r['lopo']['objects_probe_better']}/{r['n_objects']}) | TEST probe {r['test']['probe']:.2f} vs mean "
              f"{r['test']['guess_mean']:.2f}", flush=True)

    # 3. head tilt
    M, Mt = A["base"], T["base"]
    fo, lams_ = lopo_reg("up", A["up"], M, ang_err, "deg")
    lam = max(set(lams_), key=lams_.count)
    tr = test_reg(A["up"], M, T["up"], Mt, ang_err, lam)
    rep["reg"]["head_tilt"] = dict(unit="deg", folds=fo,
                                  lopo=dict(probe=float(np.mean([f["probe"] for f in fo.values()])),
                                            guess_mean=float(np.mean([f["guess_mean"] for f in fo.values()]))), test=tr)
    r = rep["reg"]["head_tilt"]
    print(f"head tilt: LOPO probe {r['lopo']['probe']:.2f} vs mean {r['lopo']['guess_mean']:.2f} deg | "
          f"TEST probe {tr['probe']:.2f} vs mean {tr['guess_mean']:.2f}", flush=True)

    # 2. yes/no
    tasks = {}
    for k, h in enumerate("LR"):
        tasks[f"inview_{h}"] = (A["w_in"][:, k], A["base"] & A["hav"][:, k], T["w_in"][:, k], T["base"] & T["hav"][:, k])
        tasks[f"fast30_{h}"] = (A["speed"][:, k] > 30, A["base"] & A["sp_ok"][:, k], T["speed"][:, k] > 30, T["base"] & T["sp_ok"][:, k])
    tasks["near5"] = (A["near"], A["near_ok"], T["near"], T["near_ok"])
    for name, (y, M, yt, Mt) in tasks.items():
        y, yt = y.astype(int), yt.astype(int)
        fo, Cs_ = {}, []
        for i, p in enumerate(plist):
            q = plist[(i + 1) % len(plist)]
            te, inn = M & (A["P"] == p), M & (A["P"] == q)
            tr_, trq = M & (A["P"] != p) & (A["P"] != q), M & (A["P"] != p)
            C, _ = logistic(A["Xn"][tr_], y[tr_], A["Xn"][inn], y[inn], CS)
            _, m = logistic(A["Xn"][trq], y[trq], A["Xn"][inn], y[inn], [C])
            maj = int(y[trq].mean() >= 0.5)
            if m is None or te.sum() == 0:
                continue
            res = cls_metrics(y[te], m.decision_function(A["Xn"][te]), m.predict(A["Xn"][te]))
            res["majority_acc"] = float((y[te] == maj).mean())
            res["C"] = C
            fo[p] = res
            Cs_.append(C)
        C = max(set(Cs_), key=Cs_.count)
        _, m = logistic(A["Xn"][M], y[M], A["Xn"][M], y[M], [C])
        maj = int(y[M].mean() >= 0.5)
        tres = cls_metrics(yt[Mt], m.decision_function(T["Xn"][Mt]), m.predict(T["Xn"][Mt]))
        tres["majority_acc"] = float((yt[Mt] == maj).mean())
        avg = {k_: float(np.nanmean([f[k_] for f in fo.values()])) for k_ in ["acc", "bal_acc", "auc", "majority_acc", "pos_rate"]}
        rep["cls"][name] = dict(folds=fo, lopo=avg, test=tres, majority_answer=maj)
        print(f"{name}: yes-rate {avg['pos_rate']:.3f} | LOPO acc {avg['acc']:.3f} vs majority {avg['majority_acc']:.3f}, "
              f"bal.acc {avg['bal_acc']:.3f}, AUC {avg['auc']:.3f} | TEST acc {tres['acc']:.3f} vs majority "
              f"{tres['majority_acc']:.3f}, bal.acc {tres['bal_acc']:.3f}, AUC {tres['auc']:.3f}", flush=True)
    json.dump(rep, open(f"{a.emb_dir}/probe_more.json", "w"), indent=1)
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        regs = ["objects_3d", "objects_2d", "head_tilt"]
        for j, nm in enumerate(regs):
            r = rep["reg"][nm]
            ax[0].bar([j - 0.2, j + 0.2], [r["lopo"]["guess_mean"] / r["lopo"]["guess_mean"], r["lopo"]["probe"] / r["lopo"]["guess_mean"]],
                      0.4, color=["gray", "tab:blue"], label=None if j else "guess-average / probe")
            ax[0].text(j + 0.2, r["lopo"]["probe"] / r["lopo"]["guess_mean"] + 0.02,
                       f"{r['lopo']['probe']:.1f} vs {r['lopo']['guess_mean']:.1f} {r['unit']}", ha="center", fontsize=7)
        ax[0].axhline(1, color="k", lw=0.6)
        ax[0].set_xticks(range(3), ["objects 3D", "objects 2D", "head tilt"])
        ax[0].set_ylabel("LOPO error / guess-the-average")
        ax[0].set_title("Regression probes, leave-one-participant-out", fontsize=9)
        names = list(rep["cls"])
        xs = np.arange(len(names))
        ax[1].bar(xs - 0.2, [rep["cls"][n]["lopo"]["majority_acc"] for n in names], 0.4, color="gray", label="always majority")
        ax[1].bar(xs + 0.2, [rep["cls"][n]["lopo"]["acc"] for n in names], 0.4, color="tab:blue", label="logistic probe")
        ax[1].set_xticks(xs, names, fontsize=7); ax[1].set_ylim(0, 1); ax[1].set_ylabel("accuracy (LOPO)")
        ax[1].legend(fontsize=7); ax[1].set_title("Yes/no probes: accuracy", fontsize=9)
        ax[2].bar(xs, [rep["cls"][n]["lopo"]["auc"] for n in names], 0.5, color="tab:blue")
        ax[2].axhline(0.5, color="gray", ls="--", label="chance / majority (0.5)")
        ax[2].set_xticks(xs, names, fontsize=7); ax[2].set_ylim(0.3, 1); ax[2].set_ylabel("AUC (LOPO)")
        ax[2].legend(fontsize=7); ax[2].set_title("Yes/no probes: AUC", fontsize=9)
        fig.suptitle(f"Probes on z, {os.path.basename(a.emb_dir.rstrip('/'))}, LOPO over the 7 training participants "
                     "(test not shown: report only)", fontsize=9)
        fig.tight_layout()
        fig.savefig(a.fig, dpi=110)
        print("SAVED", a.fig)
    print("PROBE MORE OK")


if __name__ == "__main__":
    main()
