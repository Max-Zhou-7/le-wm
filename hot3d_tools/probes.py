"""Linear probes on frozen per-frame embeddings (from encode.py).

Targets (per frame, CAMERA frame of that frame, RGB 214-1 not rotated, x right / y down / z fwd):
  wrist   [6]   left xyz, right xyz (cm)           valid: hand_valid & head_valid
  obj     [O*3] each object's position (cm)        valid: obj_valid & head_valid (one probe per object)
  up      [3]   world "up" direction (unit vector) valid: head_valid; error = angle in degrees
                (the up axis of the HOT3D world frame is set by --up; check with actions.py stats)
Rows whose target is invalid are never used (RULE: 0 poses are not real poses).
qa_pass filtering is OFF by default (--qa only after the user approves it).

Ridge regression on standardized features; lambda chosen on val (per target group), then
refit on train and reported on test. Weights saved for scoring forecasts later.

Usage (lewm env, CPU is fine):
  python hot3d_tools/probes.py <emb_dir> <lance_dir> [--fov 110] [--up z] [--qa]
Output: <emb_dir>/probes.npz, <emb_dir>/probe_report.json
"""
import argparse
import json

import numpy as np

from actions import pose7_to_Rt

LAMBDAS = [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]
REL_CM = 5.0  # "reliable" threshold from the proposal (e.g. 5 cm)


def targets(lance_dir, split, fov, rows, up_axis, qa):
    import lance
    ds = lance.dataset(f"{lance_dir}/hot3d_{split}_fov{fov}.lance")
    cols = ["T_world_device", "T_device_camera", "head_valid", "hand_wrist", "hand_valid", "obj_pose",
            "obj_valid", "qa_pass"]
    tab = ds.take(rows.tolist(), columns=cols)
    g = lambda c: np.asarray(tab[c].to_pylist(), np.float64)
    N = len(rows)
    Rwd, twd = pose7_to_Rt(g("T_world_device"))
    Rdc, tdc = pose7_to_Rt(g("T_device_camera"))
    Rwc = Rwd @ Rdc
    twc = np.einsum("nij,nj->ni", Rwd, tdc) + twd
    RcT = np.swapaxes(Rwc, -1, -2)
    to_cam = lambda p: np.einsum("nij,nj->ni", RcT, p - twc) * 100.0
    hv = g("head_valid").reshape(N) > 0.5
    keep = np.ones(N, bool) if not qa else g("qa_pass").reshape(N) > 0.5
    hw = g("hand_wrist").reshape(N, 2, 7)
    hav = g("hand_valid").reshape(N, 2) > 0.5
    wrist = np.concatenate([to_cam(hw[:, 0, :3]), to_cam(hw[:, 1, :3])], 1)
    wrist_ok = np.stack([hav[:, 0], hav[:, 1]], 1) & hv[:, None] & keep[:, None]
    op = g("obj_pose").reshape(N, -1, 7)
    ov = g("obj_valid").reshape(N, -1) > 0.5
    obj = np.stack([to_cam(op[:, o, :3]) for o in range(op.shape[1])], 1)  # [N,O,3]
    obj_ok = ov & hv[:, None] & keep[:, None]
    e_up = np.zeros(3); e_up["xyz".index(up_axis.lstrip("-"))] = -1.0 if up_axis.startswith("-") else 1.0
    up = np.einsum("nij,j->ni", RcT, e_up)
    return dict(wrist=wrist, wrist_ok=wrist_ok, obj=obj, obj_ok=obj_ok, up=up, up_ok=hv & keep)


def ridge(X, Y, lam):
    D = X.shape[1]
    A = X.T @ X + lam * len(X) * np.eye(D)
    return np.linalg.solve(A, X.T @ Y)


def add_bias(X):
    return np.concatenate([X, np.ones((len(X), 1))], 1)


def fit_select(Xtr, Ytr, Xva, Yva, err_fn):
    best = None
    for lam in LAMBDAS:
        W = ridge(Xtr, Ytr, lam)
        e = err_fn(Xva @ W, Yva).mean()
        if best is None or e < best[0]:
            best = (e, lam, W)
    return best


def wrist_err(P, Y):
    return np.linalg.norm(P - Y, axis=-1)


def ang_err(P, Y):
    P = P / np.linalg.norm(P, axis=-1, keepdims=True).clip(1e-9)
    return np.degrees(np.arccos((P * Y).sum(-1).clip(-1, 1)))


def summary(e):
    e = np.asarray(e)
    if len(e) == 0:
        return dict(n=0)
    return dict(n=int(len(e)), mean=float(e.mean()), median=float(np.median(e)),
                p90=float(np.percentile(e, 90)), frac_lt_5=float((e < REL_CM).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emb_dir")
    ap.add_argument("lance_dir")
    ap.add_argument("--fov", type=int, default=110)
    ap.add_argument("--up", default="z", help="world up axis, e.g. z, -y")
    ap.add_argument("--qa", action="store_true")
    ap.add_argument("--min-obj-frames", type=int, default=500)
    a = ap.parse_args()
    S = {}
    for split in ["train", "val", "test"]:
        E = np.load(f"{a.emb_dir}/emb_{split}.npy").astype(np.float64)
        rows = np.load(f"{a.emb_dir}/rows_{split}.npy")
        S[split] = dict(X=E, **targets(a.lance_dir, split, a.fov, rows, a.up, a.qa))
        print(f"{split}: {len(E)} frames", flush=True)
    mu, sd = S["train"]["X"].mean(0), S["train"]["X"].std(0) + 1e-6
    for s in S.values():
        s["X"] = add_bias((s["X"] - mu) / sd)
    rep, W = dict(emb_dir=a.emb_dir, up=a.up, qa=a.qa), dict(mu=mu, sd=sd)

    # wrists: one probe per hand (own masks)
    for k, name in enumerate(["left", "right"]):
        m = {sp: S[sp]["wrist_ok"][:, k] for sp in S}
        y = {sp: S[sp]["wrist"][:, 3 * k:3 * k + 3] for sp in S}
        _, lam, _ = fit_select(S["train"]["X"][m["train"]], y["train"][m["train"]],
                               S["val"]["X"][m["val"]], y["val"][m["val"]], wrist_err)
        w = ridge(S["train"]["X"][m["train"]], y["train"][m["train"]], lam)
        e = wrist_err(S["test"]["X"][m["test"]] @ w, y["test"][m["test"]])
        # floor: predict the train mean position
        e0 = wrist_err(np.broadcast_to(y["train"][m["train"]].mean(0), y["test"][m["test"]].shape),
                       y["test"][m["test"]])
        rep[f"wrist_{name}_cm"] = dict(lam=lam, test=summary(e), mean_baseline=summary(e0))
        W[f"wrist_{name}"] = w

    # objects: one probe per object seen often enough in train
    obj_rep, all_e, all_e0 = {}, [], []
    O = S["train"]["obj"].shape[1]
    W["obj"] = np.zeros((O, S["train"]["X"].shape[1], 3))
    W["obj_fitted"] = np.zeros(O, bool)
    for o in range(O):
        m = {sp: S[sp]["obj_ok"][:, o] for sp in S}
        if m["train"].sum() < a.min_obj_frames or m["val"].sum() < 20:
            continue
        y = {sp: S[sp]["obj"][:, o] for sp in S}
        _, lam, _ = fit_select(S["train"]["X"][m["train"]], y["train"][m["train"]],
                               S["val"]["X"][m["val"]], y["val"][m["val"]], wrist_err)
        w = ridge(S["train"]["X"][m["train"]], y["train"][m["train"]], lam)
        W["obj"][o], W["obj_fitted"][o] = w, True
        if m["test"].sum() == 0:
            continue
        e = wrist_err(S["test"]["X"][m["test"]] @ w, y["test"][m["test"]])
        e0 = wrist_err(np.broadcast_to(y["train"][m["train"]].mean(0), y["test"][m["test"]].shape),
                       y["test"][m["test"]])
        obj_rep[o] = dict(lam=lam, n_train=int(m["train"].sum()), test_mean=float(e.mean()))
        all_e.append(e); all_e0.append(e0)
    rep["obj_cm_all"] = dict(test=summary(np.concatenate(all_e) if all_e else []),
                             mean_baseline=summary(np.concatenate(all_e0) if all_e0 else []),
                             n_objects=len(obj_rep), per_object=obj_rep)

    # up direction (head tilt)
    m = {sp: S[sp]["up_ok"] for sp in S}
    _, lam, _ = fit_select(S["train"]["X"][m["train"]], S["train"]["up"][m["train"]],
                           S["val"]["X"][m["val"]], S["val"]["up"][m["val"]], ang_err)
    w = ridge(S["train"]["X"][m["train"]], S["train"]["up"][m["train"]], lam)
    e = ang_err(S["test"]["X"][m["test"]] @ w, S["test"]["up"][m["test"]])
    e0 = ang_err(np.broadcast_to(S["train"]["up"][m["train"]].mean(0), S["test"]["up"][m["test"]].shape),
                 S["test"]["up"][m["test"]])
    rep["up_deg"] = dict(lam=lam, test=summary(e), mean_baseline=summary(e0))
    W["up"] = w

    np.savez(f"{a.emb_dir}/probes.npz", **W)
    json.dump(rep, open(f"{a.emb_dir}/probe_report.json", "w"), indent=1)
    for k, v in rep.items():
        if isinstance(v, dict) and "test" in v:
            t, b = v["test"], v["mean_baseline"]
            print(f"{k:16s} test mean {t.get('mean', float('nan')):6.2f} median {t.get('median', float('nan')):6.2f}"
                  f" <5: {t.get('frac_lt_5', float('nan')):.2f} (n={t['n']}) | predict-mean floor "
                  f"{b.get('mean', float('nan')):6.2f}", flush=True)
    print("PROBES OK")


if __name__ == "__main__":
    main()
