"""Wrist-probe diagnosis and fix. Probes read z = emb_<split>.npy (encoder + projector output,
192 numbers, the space the predictor forecasts in).

Frame mask ("clean"): wrist valid & head valid & qa_pass & wrist inside the image
(z > 0, projects into the 224x224 pinhole frame with K, and within the 1 rad lens limit,
i.e. not in the black corners).

  diag <emb_dir> <lance_dir>   step 1: linear probe error on train / val / test vs guess-the-average,
                               clean mask and (for comparison) the old mask (valid only).
                               Also re-encodes 16 val rows to check emb/rows alignment.
  fix  <emb_dir> <lance_dir>   step 2 (VAL ONLY, never loads test): ridge with standardized targets
                               (lambda on val), 1-hidden-layer MLP (early stopping on val), and 2D
                               (pixel) probes, each vs guess-the-average.
Output: <emb_dir>/probe_diag.json or probe_fix.json (+ figure with --fig).
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from probes import targets, ridge, add_bias  # noqa: E402

LAMS = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
LENS = 1.0  # rad, projectaria max_solid_angle
IMG = 224


def load(emb_dir, lance_dir, split):
    import lance
    X = np.load(f"{emb_dir}/emb_{split}.npy").astype(np.float64)
    rows = np.load(f"{emb_dir}/rows_{split}.npy")
    T = targets(lance_dir, split, 110, rows, "z", qa=False)
    ds = lance.dataset(f"{lance_dir}/hot3d_{split}_fov110.lance")
    qa = np.asarray(ds.take(rows.tolist(), columns=["qa_pass"])["qa_pass"].to_pylist(), np.float64).reshape(-1) > 0.5
    K = np.asarray(json.load(open(f"{lance_dir}/hot3d_fov110_meta.json"))["K"])
    W = T["wrist"].reshape(-1, 2, 3)                                  # cm, camera frame
    z = W[..., 2]
    zs = np.where(z > 1e-6, z, 1.0)
    uv = np.stack([K[0, 0] * W[..., 0] / zs + K[0, 2], K[1, 1] * W[..., 1] / zs + K[1, 2]], -1)  # [N,2,2] px
    ang = np.arccos(np.clip(z / np.linalg.norm(W, axis=-1).clip(1e-9), -1, 1))
    inside = (z > 1e-6) & (uv[..., 0] >= 0) & (uv[..., 0] < IMG) & (uv[..., 1] >= 0) & (uv[..., 1] < IMG) & (ang < LENS)
    ok = T["wrist_ok"]
    return dict(X=X, rows=rows, W=W, uv=uv, ok_old=ok, clean=ok & qa[:, None] & inside,
                qa=qa, inside=inside)


def err(P, Y):
    return np.linalg.norm(P - Y, axis=-1)


def summ(e):
    return dict(n=int(len(e)), mean=float(e.mean()), median=float(np.median(e))) if len(e) else dict(n=0)


def fit_ridge(Xtr, Ytr, Xva, Yva, lams=LAMS, std_y=True):
    """Ridge on standardized X (with bias) and (optionally) standardized Y; lambda on val (mean error in
    original units)."""
    ym, ys = (Ytr.mean(0), Ytr.std(0) + 1e-9) if std_y else (0.0, 1.0)
    best = None
    for lam in lams:
        Wt = ridge(Xtr, (Ytr - ym) / ys, lam)
        e = err(Xva @ Wt * ys + ym, Yva).mean()
        if best is None or e < best[0]:
            best = (e, lam, Wt)
    _, lam, Wt = best
    return lam, (lambda X: X @ Wt * ys + ym)


def fit_mlp(Xtr, Ytr, Xva, Yva, hidden=256, lr=1e-3, wd=1e-4, epochs=300, patience=15, seed=0, fixed_epochs=0):
    """Early stopping on (Xva, Yva); or, with fixed_epochs > 0, train exactly that many epochs (no val)."""
    import torch
    torch.manual_seed(seed)
    if fixed_epochs:
        Xva, Yva, epochs, patience = Xtr[:2000], Ytr[:2000], fixed_epochs, 10 ** 9
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-9
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor((Ytr - ym) / ys, dtype=torch.float32)
    xv = torch.tensor(Xva, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, Ytr.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    pred = lambda X: net(torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X).detach().numpy() * ys + ym
    best, state, bad, hist = 1e9, None, 0, []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for b in range(0, len(xt), 512):
            i = perm[b:b + 512]
            loss = ((net(xt[i]) - yt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            ev = err(pred(xv), Yva).mean()
            et = err(pred(xt[:20000]), Ytr[:20000]).mean()
        hist.append((ep, float(et), float(ev)))
        if ev < best - 1e-4 or fixed_epochs:
            best, bad = ev, 0
            state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        return (lambda X: pred(X)), hist


def realign_check(emb_dir, lance_dir, S):
    """Re-encode 16 val rows and compare to the stored embeddings."""
    import torch
    from torchvision.io import decode_jpeg, ImageReadMode
    import lance
    from encode import load_model, MEAN, STD
    pol = json.load(open(f"{emb_dir}/info.json"))["policy"]
    m = load_model(pol)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(S["rows"]), 16, replace=False))
    ds = lance.dataset(f"{lance_dir}/hot3d_val_fov110.lance")
    blobs = ds.take(S["rows"][idx].tolist(), columns=["pixels"])["pixels"].to_pylist()
    x = torch.stack(decode_jpeg([torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in blobs], mode=ImageReadMode.RGB))
    with torch.no_grad():
        e = m.encode({"pixels": ((x.float() / 255 - MEAN) / STD)[:, None]})["emb"][:, 0].numpy()
    d_same = np.linalg.norm(e - S["X"][idx], axis=1)
    d_next = np.linalg.norm(e - S["X"][np.minimum(idx + 1, len(S["X"]) - 1)], axis=1)
    return dict(policy=pol, dist_same_row=float(d_same.mean()), dist_next_row=float(d_next.mean()),
                emb_norm=float(np.linalg.norm(e, axis=1).mean()))


def diag(a):
    S = {sp: load(a.emb_dir, a.lance_dir, sp) for sp in ["train", "val", "test"]}
    mu, sd = S["train"]["X"].mean(0), S["train"]["X"].std(0) + 1e-6
    for s in S.values():
        s["Xs"] = add_bias((s["X"] - mu) / sd)
    rep = dict(emb_dir=a.emb_dir, dim=int(S["train"]["X"].shape[1]))
    for sp, s in S.items():
        rep[f"mask_{sp}"] = {h: dict(valid=int(s["ok_old"][:, k].sum()), qa=int((s["ok_old"][:, k] & s["qa"]).sum()),
                                     inside=int((s["ok_old"][:, k] & s["inside"][:, k]).sum()),
                                     clean=int(s["clean"][:, k].sum()), n=len(s["X"]))
                             for k, h in enumerate("LR")}
    for mask in ["clean", "ok_old"]:
        for k, h in enumerate("LR"):
            M = {sp: S[sp][mask][:, k] for sp in S}
            Y = {sp: S[sp]["W"][M[sp], k] for sp in S}
            Xm = {sp: S[sp]["Xs"][M[sp]] for sp in S}
            mean = Y["train"].mean(0)
            r = {}
            for name, lams in [("ols", [1e-8]), ("ridge_val_lam", LAMS)]:
                lam, f = fit_ridge(Xm["train"], Y["train"], Xm["val"], Y["val"], lams, std_y=False)
                r[name] = dict(lam=lam, **{sp: summ(err(f(Xm[sp]), Y[sp])) for sp in S})
                r[name]["per_axis_val_cm"] = np.abs(f(Xm["val"]) - Y["val"]).mean(0).round(2).tolist()
            r["guess_mean"] = {sp: summ(err(mean, Y[sp])) for sp in S}
            r["guess_mean"]["per_axis_val_cm"] = np.abs(mean - Y["val"]).mean(0).round(2).tolist()
            # how much the average itself moves between splits
            r["split_means_cm"] = {sp: Y[sp].mean(0).round(1).tolist() for sp in S}
            rep[f"{mask}_{h}"] = r
    rep["realign"] = realign_check(a.emb_dir, a.lance_dir, S["val"])
    json.dump(rep, open(f"{a.emb_dir}/probe_diag.json", "w"), indent=1)
    for mask in ["clean", "ok_old"]:
        print(f"== mask {mask}  (mean 3D error, cm)")
        print(f"{'':6s}{'':16s}{'train':>8s}{'val':>8s}{'test':>8s}")
        for h in "LR":
            r = rep[f"{mask}_{h}"]
            for name in ["ols", "ridge_val_lam", "guess_mean"]:
                lam = f" lam={r[name]['lam']:g}" if "lam" in r[name] else ""
                print(f"{h:6s}{name:16s}" + "".join(f"{r[name][sp].get('mean', float('nan')):8.2f}" for sp in S) + lam)
            print(f"{h:6s}per-axis val |err| x,y,z: ridge {r['ridge_val_lam']['per_axis_val_cm']}  mean {r['guess_mean']['per_axis_val_cm']}")
            print(f"{h:6s}split means: {r['split_means_cm']}")
    for sp in S:
        print("frames", sp, rep[f"mask_{sp}"])
    print("realign", rep["realign"])
    print("DIAG OK")


def fix(a):
    S = {sp: load(a.emb_dir, a.lance_dir, sp) for sp in ["train", "val"]}     # test NOT loaded
    mu, sd = S["train"]["X"].mean(0), S["train"]["X"].std(0) + 1e-6
    for s in S.values():
        s["Xs"] = add_bias((s["X"] - mu) / sd)
        s["Xn"] = (s["X"] - mu) / sd
    rep, hists = dict(emb_dir=a.emb_dir, split="val", mask="clean"), {}
    lams = [10.0 ** p for p in range(-6, 5)]
    for k, h in enumerate("LR"):
        M = {sp: S[sp]["clean"][:, k] for sp in S}
        for tgt, key, unit in [("3d", "W", "cm"), ("2d", "uv", "px")]:
            Y = {sp: S[sp][key][M[sp], k] for sp in S}
            r = dict(unit=unit, n_train=int(M["train"].sum()), n_val=int(M["val"].sum()))
            mean = Y["train"].mean(0)
            r["guess_mean"] = dict(val=summ(err(mean, Y["val"])), train=summ(err(mean, Y["train"])))
            lam, f = fit_ridge(S["train"]["Xs"][M["train"]], Y["train"], S["val"]["Xs"][M["val"]], Y["val"], lams)
            r["linear"] = dict(lam=lam, val=summ(err(f(S["val"]["Xs"][M["val"]]), Y["val"])),
                               train=summ(err(f(S["train"]["Xs"][M["train"]]), Y["train"])))
            g, hist = fit_mlp(S["train"]["Xn"][M["train"]], Y["train"], S["val"]["Xn"][M["val"]], Y["val"])
            r["mlp"] = dict(hidden=256, epochs=len(hist), best_epoch=int(np.argmin([x[2] for x in hist])),
                            val=summ(err(g(S["val"]["Xn"][M["val"]]), Y["val"])),
                            train=summ(err(g(S["train"]["Xn"][M["train"]]), Y["train"])))
            hists[f"{h}_{tgt}"] = hist
            rep[f"{h}_{tgt}"] = r
            print(f"{h} {tgt} ({unit}) n_train {r['n_train']} n_val {r['n_val']}: guess-mean val {r['guess_mean']['val']['mean']:.2f} | "
                  f"linear (lam {lam:g}) train {r['linear']['train']['mean']:.2f} val {r['linear']['val']['mean']:.2f} | "
                  f"MLP train {r['mlp']['train']['mean']:.2f} val {r['mlp']['val']['mean']:.2f} "
                  f"(best epoch {r['mlp']['best_epoch']})", flush=True)
    rep["mlp_hist"] = hists
    json.dump(rep, open(f"{a.emb_dir}/probe_fix.json", "w"), indent=1)
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        names = ["guess_mean", "linear", "mlp"]
        for j, tgt in enumerate(["3d", "2d"]):
            xs = np.arange(2)
            for i, nm in enumerate(names):
                ax[j].bar(xs + (i - 1) * 0.27, [rep[f"{h}_{tgt}"][nm]["val"]["mean"] for h in "LR"], 0.27, label=nm)
            ax[j].set_xticks(xs, ["left wrist", "right wrist"])
            ax[j].set_ylabel("val mean error (" + ("cm, 3D camera frame" if tgt == "3d" else "px, 224 image") + ")")
            ax[j].set_title(f"{tgt.upper()} wrist probe, val (clean frames)")
            ax[j].legend(fontsize=8)
        for key, hist in hists.items():
            h = np.array(hist)
            ax[2].plot(h[:, 0], h[:, 2] / rep[key]["guess_mean"]["val"]["mean"], label=f"{key} val")
            ax[2].plot(h[:, 0], h[:, 1] / rep[key]["guess_mean"]["train"]["mean"], ls=":", label=f"{key} train")
        ax[2].axhline(1, color="k", lw=0.7)
        ax[2].set_xlabel("MLP epoch"); ax[2].set_ylabel("error / guess-the-average")
        ax[2].set_title("MLP training (early stop on val)"); ax[2].legend(fontsize=6, ncol=2)
        fig.suptitle(f"Wrist probes on z (192-d), {os.path.basename(a.emb_dir.rstrip('/'))}. "
                     "Clean = valid & qa_pass & inside image. Val only.", fontsize=9)
        fig.tight_layout()
        fig.savefig(a.fig, dpi=110)
        print("SAVED", a.fig)
    print("FIX OK")


def participants(lance_dir, split, rows):
    import lance
    ds = lance.dataset(f"{lance_dir}/hot3d_{split}_fov110.lance")
    rec = np.asarray(ds.take(rows.tolist(), columns=["recording"])["recording"].to_pylist())
    return np.array([r.split("_")[0] for r in rec])


def lopo(a):
    """Leave-one-participant-out over the 7 GT training participants (train + val tables; val
    subsampled to stride 3 like train). Fold p: test on p; inner participant q (next in the list)
    picks lambda / MLP early stopping; fit on the other 5. Then, for reporting only: fit on all 7
    (lambda = the most common fold choice, MLP epochs = median fold best epoch) and score test,
    split into a constant offset per test participant vs the remainder. Test chooses nothing."""
    S = {sp: load(a.emb_dir, a.lance_dir, sp) for sp in ["train", "val", "test"]}
    for sp, s in S.items():
        s["P"] = participants(a.lance_dir, sp, s["rows"])
    v = S["val"]
    keep = np.flatnonzero(v["rows"] % 3 == 0)
    A = {k: (np.concatenate([S["train"][k], v[k][keep]]) if isinstance(v[k], np.ndarray) and len(v[k]) == len(v["X"])
             else S["train"][k]) for k in S["train"]}
    T = S["test"]
    mu, sd = A["X"].mean(0), A["X"].std(0) + 1e-6
    for s in (A, T):
        s["Xn"] = (s["X"] - mu) / sd
        s["Xs"] = add_bias(s["Xn"])
    plist = sorted(set(A["P"]))
    lams = [10.0 ** p for p in range(-6, 5)]
    rep = dict(emb_dir=a.emb_dir, participants=plist, mask="clean", folds={}, test={})
    for k, h in enumerate("LR"):
        for tgt, key, unit in [("3d", "W", "cm"), ("2d", "uv", "px")]:
            name = f"{h}_{tgt}"
            Y = A[key][:, k]
            M = A["clean"][:, k]
            folds, lam_pick, ep_pick = {}, [], []
            for i, p in enumerate(plist):
                q = plist[(i + 1) % len(plist)]
                te, inn = M & (A["P"] == p), M & (A["P"] == q)
                tr = M & (A["P"] != p) & (A["P"] != q)
                trq = M & (A["P"] != p)                       # all 6 for the final fold fit
                mean = Y[trq].mean(0)
                lam, _ = fit_ridge(A["Xs"][tr], Y[tr], A["Xs"][inn], Y[inn], lams)
                _, f = fit_ridge(A["Xs"][trq], Y[trq], A["Xs"][inn], Y[inn], [lam])   # refit on all 6
                g, hist = fit_mlp(A["Xn"][tr], Y[tr], A["Xn"][inn], Y[inn])
                be = int(np.argmin([x[2] for x in hist])) + 1
                lam_pick.append(lam); ep_pick.append(be)
                folds[p] = dict(n=int(te.sum()), guess_mean=float(err(mean, Y[te]).mean()),
                                linear=float(err(f(A["Xs"][te]), Y[te]).mean()),
                                mlp=float(err(g(A["Xn"][te]), Y[te]).mean()), lam=lam, mlp_best_epoch=be)
                print(f"{name} fold {p} (inner {q}) n {te.sum()}: mean {folds[p]['guess_mean']:.2f} "
                      f"linear {folds[p]['linear']:.2f} (lam {lam:g}) mlp {folds[p]['mlp']:.2f} (ep {be})", flush=True)
            avg = {m: float(np.mean([f_[m] for f_ in folds.values()])) for m in ["guess_mean", "linear", "mlp"]}
            rep["folds"][name] = dict(unit=unit, per_participant=folds, avg_over_participants=avg)
            # reporting only: fit on all 7, score test, offset vs remainder per test participant
            lam = max(set(lam_pick), key=lam_pick.count)
            ep = int(np.median(ep_pick))
            _, f = fit_ridge(A["Xs"][M], Y[M], A["Xs"][M][:10], Y[M][:10], [lam])
            g, _ = fit_mlp(A["Xn"][M], Y[M], None, None, fixed_epochs=ep)
            mean = Y[M].mean(0)
            Mt, Yt = T["clean"][:, k], T[key][:, k]
            preds = dict(guess_mean=np.broadcast_to(mean, Yt.shape), linear=f(T["Xs"]), mlp=g(T["Xn"]))
            trep = dict(lam=lam, mlp_epochs=ep)
            for m, P in preds.items():
                tot, per = err(P[Mt], Yt[Mt]).mean(), {}
                rem_all = []
                for p in sorted(set(T["P"][Mt])):
                    mm = Mt & (T["P"] == p)
                    res = P[mm] - Yt[mm]
                    off = res.mean(0)
                    rem = err(res - off, 0)
                    rem_all.append(rem)
                    per[p] = dict(n=int(mm.sum()), total=float(err(res, 0).mean()), offset=off.round(2).tolist(),
                                  offset_norm=float(np.linalg.norm(off)), remainder=float(rem.mean()))
                trep[m] = dict(total=float(tot), remainder=float(np.concatenate(rem_all).mean()), per_participant=per)
            rep["test"][name] = trep
            print(f"{name} LOPO avg: mean {avg['guess_mean']:.2f} linear {avg['linear']:.2f} mlp {avg['mlp']:.2f} {unit}", flush=True)
            for m in preds:
                t = trep[m]
                print(f"{name} TEST {m:10s} total {t['total']:.2f} remainder {t['remainder']:.2f} | offsets " +
                      " ".join(f"{p}: {v_['offset']} (|{v_['offset_norm']:.1f}|)" for p, v_ in t["per_participant"].items()),
                      flush=True)
    json.dump(rep, open(f"{a.emb_dir}/probe_lopo.json", "w"), indent=1)
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = ["L_3d", "R_3d", "L_2d", "R_2d"]
        fig, ax = plt.subplots(2, 4, figsize=(18, 7.5))
        for j, nm in enumerate(names):
            fo = rep["folds"][nm]["per_participant"]
            xs = np.arange(len(plist))
            for i, m in enumerate(["guess_mean", "linear", "mlp"]):
                ax[0, j].bar(xs + (i - 1) * 0.27, [fo[p][m] for p in plist], 0.27, label=m)
            ax[0, j].set_xticks(xs, plist, fontsize=7)
            ax[0, j].set_title(f"{nm} leave-one-participant-out ({rep['folds'][nm]['unit']})", fontsize=9)
            ax[0, j].legend(fontsize=7)
            t = rep["test"][nm]
            tps = list(t["linear"]["per_participant"])
            xs = np.arange(len(tps))
            for i, m in enumerate(["guess_mean", "linear", "mlp"]):
                tot = [t[m]["per_participant"][p]["total"] for p in tps]
                rem = [t[m]["per_participant"][p]["remainder"] for p in tps]
                ax[1, j].bar(xs + (i - 1) * 0.27, tot, 0.27, alpha=0.35, color=f"C{i}")
                ax[1, j].bar(xs + (i - 1) * 0.27, rem, 0.27, color=f"C{i}", label=f"{m} (solid = after removing offset)")
            ax[1, j].set_xticks(xs, tps)
            ax[1, j].set_title(f"{nm} TEST (report only): total (light) vs remainder (solid)", fontsize=9)
            ax[1, j].legend(fontsize=6)
        fig.suptitle("Wrist probes on z, clean frames. Top: LOPO over the 7 training participants (choice). "
                     "Bottom: test, offset per participant vs remainder (not used to choose).", fontsize=9)
        fig.tight_layout()
        fig.savefig(a.fig, dpi=100)
        print("SAVED", a.fig)
    print("LOPO OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diag", "fix", "lopo"])
    ap.add_argument("emb_dir")
    ap.add_argument("lance_dir")
    ap.add_argument("--fig", default="")
    a = ap.parse_args()
    {"diag": diag, "fix": fix, "lopo": lopo}[a.cmd](a)
