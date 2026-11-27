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


def fit_mlp(Xtr, Ytr, Xva, Yva, hidden=256, lr=1e-3, wd=1e-4, epochs=300, patience=15, seed=0):
    import torch
    torch.manual_seed(seed)
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
        if ev < best - 1e-4:
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diag", "fix"])
    ap.add_argument("emb_dir")
    ap.add_argument("lance_dir")
    ap.add_argument("--fig", default="")
    a = ap.parse_args()
    diag(a) if a.cmd == "diag" else fix(a)
