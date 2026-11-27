"""Gap-forecasting evaluation (RQ1): predict the frame after a gap of g steps at 10 fps.

Window (raw 30 fps rows of one recording): anchor row r = last observed frame.
  history  r-27, r-24, ..., r      (10 obs = 1 s at 10 fps; LeWM / B1 use the last 3)
  targets  r+3g, g in {1,2,4,8,16,32}    (100, 200, 400, 800, 1600, 3200 ms)
Windows whose rows [r-27, r+96] contain a step > 50 ms are skipped (windows.py rule), so every gap
is scored on the same windows.
Anchors every --every rows (default 15 = 0.5 s).

Methods (embedding methods are scored through the frozen probes from probes.py):
  B0  repeat      z_hat = z_r
  B1  linear      z_hat = z_r + g (z_r - z_{r-3})
  B2  timewarp    warp the ORIGINAL fisheye at r into the pinhole view of r+3g using the
                  true head rotation (head pose is known during the gap: SLAM/IMU), encode.
                  Directions the old fisheye never saw stay black (the sensor image ends at ~57 deg
                  off-axis, the pinhole needs 55 deg at the sides and 63.7 at the corners).
      timewarp_fill  same, but uncovered pixels take the unwarped last frame
  B3  kalman_gt   constant-velocity Kalman per wrist on the GT wrist track (world frame),
                  predicted position put into camera(r+3g) with the true head pose
      kalman_probe  same, observations = probe(z) of the history frames
      hold_gt     last GT wrist position (world), no motion
  oracle          probe(z_{r+3g}) = the probe's own error on the real frame
  M_lewm_<variant>  LeWM rollout (lewm_forecast.py; --lewm <policy> --variant <v>)
  M_lewm_<variant>_noise<s>  same, but the wrist-motion inputs are recomputed from wrist positions with
                  Gaussian noise (sigma s cm per axis, per raw frame); --wrist-noise 0.5 1 2. No retraining.
Kalman q, r: picked per gap on the eval split (grid) unless --kalman-params <json> gives them
(test MUST use the values frozen on val: {"kalman_gt": {"g1": [q, r], ...}, "kalman_probe": {...}}).

Scores per method and gap (only where the target is valid at r+3g; 0 poses never used):
  wrist_cm (left+right pooled, mean / median / frac < 5 cm), obj_cm, up_deg,
  wrist_above_floor = mean over the same (window, hand) samples of err(method) - err(oracle), where
      oracle = the probe on the real target frame (the probe's own floor). For GT-based methods
      (hold_gt, kalman_gt) that do not go through the probe, it can be negative.
  emb_err = ||z_hat - z_{r+3g}|| / mean ||z - mean z||  (embedding methods),
  pix_mae (timewarp vs the stored frame, /255).

Usage (lewm env):
  python hot3d_tools/gap_eval.py <emb_dir> <lance_dir> <fisheye_dir> --split val|test
      [--policy lewm/...pt] [--every 15] [--q ...] [--r ...] [--no-timewarp]
  <emb_dir> must have emb_<split>.npy with stride 1 and probes.npz (from probes.py).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from actions import pose7_to_Rt  # noqa: E402
from windows import valid_starts  # noqa: E402

GAPS = [1, 2, 4, 8, 16, 32]
HIST = 9  # history steps before r (10 obs incl. r)
FS = 3    # frameskip (10 fps)


def load_split(lance_dir, split, fov, extra=()):
    import lance
    ds = lance.dataset(f"{lance_dir}/hot3d_{split}_fov{fov}.lance")
    cols = ["episode_idx", "step_idx", "t_sec", "recording", "T_world_device", "T_device_camera", "head_valid",
            "hand_wrist", "hand_valid", "obj_pose", "obj_valid"] + list(extra)
    tab = ds.to_table(columns=cols)
    g = lambda c: np.asarray(tab[c].to_pylist(), np.float64)
    N = tab.num_rows
    Rwd, twd = pose7_to_Rt(g("T_world_device"))
    Rdc, tdc = pose7_to_Rt(g("T_device_camera"))
    D = dict(
        ds=ds, N=N,
        ep=np.asarray(tab["episode_idx"].to_pylist()).reshape(-1),
        step=np.asarray(tab["step_idx"].to_pylist()).reshape(-1),
        t=g("t_sec").reshape(-1),
        rec=np.asarray(tab["recording"].to_pylist()),
        Rwc=Rwd @ Rdc,
        twc=np.einsum("nij,nj->ni", Rwd, tdc) + twd,
        hv=g("head_valid").reshape(N) > 0.5,
        wrist_w=g("hand_wrist").reshape(N, 2, 7)[..., :3],
        hav=g("hand_valid").reshape(N, 2) > 0.5,
        obj_w=g("obj_pose").reshape(N, -1, 7)[..., :3],
        ov=g("obj_valid").reshape(N, -1) > 0.5,
    )
    for c in extra:
        D[c] = np.asarray(tab[c].to_pylist(), np.float32).reshape(N, -1)
    return D


def to_cam(D, rows, p_world):
    """p_world [n,...,3] -> camera(rows) frame, cm."""
    R = np.swapaxes(D["Rwc"][rows], -1, -2)
    t = D["twc"][rows]
    shp = p_world.shape
    p = p_world.reshape(len(rows), -1, 3) - t[:, None]
    return (np.einsum("nij,nkj->nki", R, p) * 100.0).reshape(shp)


def anchors(D, every):
    span = (HIST + max(GAPS)) * FS + 1
    out = []
    for e in np.unique(D["ep"]):
        idx = np.flatnonzero(D["ep"] == e)
        ok = valid_starts(D["t"][idx], span)
        for s in range(0, len(ok), every):
            r = idx[s] + HIST * FS
            if ok[s] and D["hv"][r - HIST * FS: r + max(GAPS) * FS + 1: FS].all():
                out.append(r)
    return np.asarray(out, np.int64)


class Probes:
    def __init__(self, path, up_axis="z"):
        P = np.load(path)
        self.P = {k: P[k] for k in P.files}

    def _x(self, z):
        X = (np.asarray(z, np.float64) - self.P["mu"]) / self.P["sd"]
        return np.concatenate([X, np.ones((len(X), 1))], 1)

    def wrist(self, z):            # [n,2,3] camera frame cm
        X = self._x(z)
        return np.stack([X @ self.P["wrist_left"], X @ self.P["wrist_right"]], 1)

    def obj(self, z):              # [n,O,3]
        return np.einsum("nd,odk->nok", self._x(z), self.P["obj"])

    def up(self, z):
        u = self._x(z) @ self.P["up"]
        return u / np.linalg.norm(u, axis=-1, keepdims=True).clip(1e-9)


def noisy_actions(D, sigma, seed=0):
    """Action table with the wrist-motion parts (6:12) recomputed from wrist positions + Gaussian noise
    (sigma cm per axis, independent per raw frame), exactly as actions.compute_actions: displacement
    row t -> t+1 rotated into camera(t). Validity flags unchanged; invalid parts stay 0.
    sigma = 0 reproduces the stored column (checked)."""
    A = D["action"].copy()
    rng = np.random.default_rng(seed)
    W = D["wrist_w"] * 100.0 + rng.normal(0.0, sigma, D["wrist_w"].shape)     # [N,2,3] world cm
    a, b = np.arange(D["N"] - 1), np.arange(1, D["N"])
    RcT = np.swapaxes(D["Rwc"][a], -1, -2)
    for k in range(2):
        d = np.einsum("tij,tj->ti", RcT, W[b, k] - W[a, k])
        ok = A[a, 13 + k] > 0.5
        A[a, 6 + 3 * k: 9 + 3 * k] = np.where(ok[:, None], d, 0.0)
    return A


def kalman_cv(obs, ok, q, r, g):
    """obs [n,H,3] (one obs per 10 fps step, last = anchor), ok [n,H]. Constant velocity,
    dt = 1 step. Returns predicted position g steps after the last step [n,3] (NaN if no obs)."""
    n, H, _ = obs.shape
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = q * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])
    out = np.full((n, 3), np.nan)
    for i in range(n):
        if not ok[i].any():
            continue
        x = None
        for h in range(H):
            if x is not None:
                x = F @ x
                P = F @ P @ F.T + Q
            if ok[i, h]:
                if x is None:
                    x = np.stack([obs[i, h], np.zeros(3)], 1).T     # [2,3]
                    P = np.diag([r, 1e3])
                    continue
                S = P[0, 0] + r
                K = P[:, 0] / S
                x = x + np.outer(K, obs[i, h] - x[0])
                P = P - np.outer(K, P[0])
        Fg = np.array([[1.0, float(g)], [0.0, 1.0]])
        out[i] = (Fg @ x)[0]
    return out


def summ(e):
    e = np.asarray(e, np.float64)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return dict(n=0)
    return dict(n=int(len(e)), mean=float(e.mean()), median=float(np.median(e)),
                frac_lt_5=float((e < 5).mean()))


def score_pose(D, probes, tgt, z_hat=None, wrist_hat=None):
    """Errors vs GT at rows tgt. Either embeddings (decoded by the probes) or direct wrist predictions."""
    res = {}
    ok_w = D["hav"][tgt] & D["hv"][tgt][:, None]
    gt_w = to_cam(D, tgt, D["wrist_w"][tgt])
    w = probes.wrist(z_hat) if z_hat is not None else wrist_hat
    e = np.linalg.norm(w - gt_w, axis=-1)
    res["wrist_cm"] = summ(e[ok_w])
    res["_we"] = np.where(ok_w, e, np.nan)           # per (window, hand), for the floor; not saved
    if z_hat is not None:
        fitted = probes.P["obj_fitted"]
        ok_o = D["ov"][tgt] & D["hv"][tgt][:, None] & fitted[None]
        gt_o = to_cam(D, tgt, D["obj_w"][tgt])
        res["obj_cm"] = summ(np.linalg.norm(probes.obj(z_hat) - gt_o, axis=-1)[ok_o])
        up_gt = np.swapaxes(D["Rwc"][tgt], -1, -2)[:, :, 2]  # world +z in camera coords
        cosang = (probes.up(z_hat) * up_gt).sum(-1).clip(-1, 1)
        res["up_deg"] = summ(np.degrees(np.arccos(cosang)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emb_dir")
    ap.add_argument("lance_dir")
    ap.add_argument("fisheye_dir")
    ap.add_argument("--split", default="val")
    ap.add_argument("--fov", type=int, default=110)
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--policy", default=None, help="encoder for TimeWarp images (same as emb_dir's)")
    ap.add_argument("--q", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--r", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--no-timewarp", action="store_true")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--lewm", default=None, help="LeWM policy for method M (same checkpoint as emb_dir)")
    ap.add_argument("--variant", default="head_hands", help="action variant the LeWM policy was trained with")
    ap.add_argument("--kalman-params", default=None, help="json with fixed (q, r) per gap (frozen on val)")
    ap.add_argument("--wrist-noise", type=float, nargs="*", default=[], help="LeWM with noisy wrist-motion inputs (cm)")
    ap.add_argument("--out", default=None, help="output json (default <emb_dir>/gap_eval_<split>.json)")
    a = ap.parse_args()
    assert a.split != "test" or a.kalman_params, "test needs --kalman-params frozen on val"
    KP = json.load(open(a.kalman_params)) if a.kalman_params else None
    if KP and "kalman_params" in KP:          # a val gap_eval json: use the values it chose
        KP = KP["kalman_params"]
    t0 = time.time()
    D = load_split(a.lance_dir, a.split, a.fov, extra=["action"] if a.lewm else [])
    Z = np.load(f"{a.emb_dir}/emb_{a.split}.npy").astype(np.float32)
    rows = np.load(f"{a.emb_dir}/rows_{a.split}.npy")
    assert len(rows) == D["N"] and (rows == np.arange(D["N"])).all(), "need stride-1 embeddings"
    probes = Probes(f"{a.emb_dir}/probes.npz")
    R = anchors(D, a.every)
    if a.max_windows:
        R = R[:a.max_windows]
    print(f"{a.split}: {len(R)} windows ({time.time() - t0:.0f}s)", flush=True)
    zscale = float(np.linalg.norm(Z - Z.mean(0), axis=1).mean())
    rep = dict(split=a.split, n_windows=int(len(R)), every=a.every, emb_dir=a.emb_dir, gaps_ms=[100 * g for g in GAPS],
               methods={})

    floor = {}

    def put(method, g, res):
        we = res.pop("_we", None)
        if we is not None:
            if method == "oracle":
                floor[g] = we
            d = we - floor[g]
            d = d[np.isfinite(d)]
            res["wrist_above_floor"] = float(d.mean()) if len(d) else float("nan")
        rep["methods"].setdefault(method, {})[f"g{g}"] = res

    hist = np.stack([R - FS * h for h in range(HIST, -1, -1)], 1)  # [n,10], last = r
    # Kalman / hold: observations in world frame (cm), GT or probe
    obs_gt = D["wrist_w"][hist] * 100.0                                    # [n,10,2,3]
    ok_gt = D["hav"][hist]                                                 # [n,10,2]
    pw_cam = probes.wrist(Z[hist.reshape(-1)]).reshape(len(R), HIST + 1, 2, 3) / 100.0
    obs_pr = (np.einsum("nhij,nhkj->nhki", D["Rwc"][hist], pw_cam) + D["twc"][hist][:, :, None]) * 100.0
    ok_pr = np.ones_like(ok_gt)

    def kalman_score(obs, ok, q, r, g, tgt):
        pred_w = np.stack([kalman_cv(obs[:, :, k], ok[:, :, k], q, r, g) for k in range(2)], 1) / 100.0
        return score_pose(D, probes, tgt, wrist_hat=to_cam(D, tgt, pred_w))

    for g in GAPS:
        tgt = R + FS * g
        z_true = Z[tgt]
        for name, zh in [("oracle", z_true), ("B0_repeat", Z[R]), ("B1_linear", Z[R] + g * (Z[R] - Z[R - FS]))]:
            res = score_pose(D, probes, tgt, z_hat=zh)
            res["emb_err"] = float(np.linalg.norm(zh - z_true, axis=1).mean() / zscale)
            put(name, g, res)
        put("hold_gt", g, kalman_score(obs_gt, ok_gt, 0.0, 1e-6, 0, tgt))
        for tag, obs, ok in [("kalman_gt", obs_gt, ok_gt), ("kalman_probe", obs_pr, ok_pr)]:
            if KP:
                best = tuple(KP[tag][f"g{g}"])
                grid = {best: kalman_score(obs, ok, best[0], best[1], g, tgt)}
            else:
                grid = {}
                for q in a.q:
                    for r in a.r:
                        grid[(q, r)] = kalman_score(obs, ok, q, r, g, tgt)
                best = min(grid, key=lambda k: grid[k]["wrist_cm"].get("mean", np.inf))
            res = dict(grid[best], q=best[0], r=best[1], params_from=a.kalman_params or f"grid on {a.split}")
            rep.setdefault("kalman_params", {}).setdefault(tag, {})[f"g{g}"] = [best[0], best[1]]
            put(tag, g, res)
        print(f"g={g} done ({time.time() - t0:.0f}s)", flush=True)

    if a.lewm:
        from lewm_forecast import Forecaster, raw_actions
        Fc = Forecaster(a.lewm, a.variant, a.lance_dir)
        K = max(GAPS)
        zf = Fc.rollout(np.stack([Z[R - 2 * FS], Z[R - FS], Z[R]], 1), raw_actions(D, R, K), K)
        for g in GAPS:
            tgt = R + FS * g
            res = score_pose(D, probes, tgt, z_hat=zf[:, g - 1])
            res["emb_err"] = float(np.linalg.norm(zf[:, g - 1] - Z[tgt], axis=1).mean() / zscale)
            put(f"M_lewm_{a.variant}", g, res)
        print(f"LeWM rollout done ({time.time() - t0:.0f}s)", flush=True)
        if a.wrist_noise:
            A0 = noisy_actions(D, 0.0)
            dev0 = float(np.abs(A0[:, 6:12] - D["action"][:, 6:12]).max())
            print(f"noise check: sigma 0 vs stored wrist actions, max diff {dev0:.2e} cm", flush=True)
            assert dev0 < 1e-2, "recomputed wrist actions do not match the stored column"
        for sig in a.wrist_noise:
            Dn = dict(D, action=noisy_actions(D, sig, seed=0))
            zf = Fc.rollout(np.stack([Z[R - 2 * FS], Z[R - FS], Z[R]], 1), raw_actions(Dn, R, K), K)
            for g in GAPS:
                tgt = R + FS * g
                res = score_pose(D, probes, tgt, z_hat=zf[:, g - 1])
                res["emb_err"] = float(np.linalg.norm(zf[:, g - 1] - Z[tgt], axis=1).mean() / zscale)
                res["wrist_noise_cm"] = sig
                put(f"M_lewm_{a.variant}_noise{sig:g}", g, res)
            print(f"LeWM rollout with wrist noise {sig} cm done ({time.time() - t0:.0f}s)", flush=True)

    if not a.no_timewarp:
        import torch
        from encode import MEAN, STD, load_model
        from fisheye import FisheyeRecording
        from timewarp import Warper
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(a.policy).to(dev)
        W = Warper(fov=a.fov)
        from torchvision.io import decode_jpeg
        cache = {}
        VARS = ["B2_timewarp", "B2_timewarp_fill"]   # black where uncovered / unwarped last frame there
        zs = {(v, g): np.zeros((len(R), Z.shape[1]), np.float32) for v in VARS for g in GAPS}
        pix = {(v, g): [] for v in VARS for g in GAPS}
        unc = {g: [] for g in GAPS}
        dec = lambda b: decode_jpeg(torch.frombuffer(bytearray(b), dtype=torch.uint8)).permute(1, 2, 0).numpy()
        for i, r in enumerate(R):
            rec = D["rec"][r]
            if rec not in cache:
                cache.clear()
                cache[rec] = FisheyeRecording(f"{a.fisheye_dir}/{rec}.npz")
            f = cache[rec]
            s = int(D["step"][r])
            assert abs(f.ts[s] / 1e9 - f.ts[0] / 1e9 - D["t"][r]) < 1e-3, "fisheye/lance timestamps differ"
            img = f.image(s)
            last = dec(D["ds"].take([int(r)], columns=["pixels"])["pixels"].to_pylist()[0])
            args = (f.d["proj_params"][s], int(f.d["full_size"]))
            kw = dict(valid_radius=f.d["valid_radius"][s], max_angle=float(f.d["max_solid_angle"][s]))
            outs = {v: [] for v in VARS}
            for g in GAPS:
                Rsd = D["Rwc"][r].T @ D["Rwc"][r + FS * g]
                o = W(img, *args, Rsd, **kw)
                outs["B2_timewarp"].append(o)
                outs["B2_timewarp_fill"].append(W(img, *args, Rsd, fill=last, **kw))
                # uncovered share of the target view's lit area
                tv = last.sum(-1) > 0
                unc[g].append(float(((o.sum(-1) == 0) & tv).sum() / max(tv.sum(), 1)))
            if i % 20 == 0:  # pixel error on a subset (decode the stored target frames)
                blobs = D["ds"].take([int(r + FS * g) for g in GAPS], columns=["pixels"])["pixels"].to_pylist()
                refs = [dec(b).astype(np.float32) for b in blobs]
                for v in VARS:
                    for g, o, ref in zip(GAPS, outs[v], refs):
                        pix[(v, g)].append(np.abs(o.astype(np.float32) - ref).mean() / 255)
            x = torch.from_numpy(np.stack(outs[VARS[0]] + outs[VARS[1]])).permute(0, 3, 1, 2)
            x = ((x.float() / 255 - MEAN) / STD).to(dev)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                e = model.encode({"pixels": x[:, None]})["emb"][:, 0].float().cpu().numpy()
            for vi, v in enumerate(VARS):
                for j, g in enumerate(GAPS):
                    zs[(v, g)][i] = e[vi * len(GAPS) + j]
            if i % 500 == 0:
                print(f"timewarp {i}/{len(R)} ({time.time() - t0:.0f}s)", flush=True)
        for v in VARS:
            for g in GAPS:
                tgt = R + FS * g
                res = score_pose(D, probes, tgt, z_hat=zs[(v, g)])
                res["emb_err"] = float(np.linalg.norm(zs[(v, g)] - Z[tgt], axis=1).mean() / zscale)
                res["pix_mae"] = float(np.mean(pix[(v, g)]))
                res["uncovered_frac"] = float(np.mean(unc[g]))
                put(v, g, res)
        # repeat-last pixel error for reference
        for g in GAPS:
            idx = R[::20]
            e = []
            for r in idx:
                b = D["ds"].take([int(r), int(r + FS * g)], columns=["pixels"])["pixels"].to_pylist()
                a0, a1 = [decode_jpeg(torch.frombuffer(bytearray(x), dtype=torch.uint8)).float() for x in b]
                e.append((a0 - a1).abs().mean().item() / 255)
            rep["methods"]["B0_repeat"][f"g{g}"]["pix_mae"] = float(np.mean(e))

    out = a.out or f"{a.emb_dir}/gap_eval_{a.split}.json"
    json.dump(rep, open(out, "w"), indent=1)
    print(f"\n{'method':14s} " + " ".join(f"{'g' + str(g) + ' wrist cm':>16s}" for g in GAPS))
    print("(mean wrist cm, + above the oracle floor)")
    for m, v in rep["methods"].items():
        cells = []
        for g in GAPS:
            x = v.get(f"g{g}", {})
            w = x.get("wrist_cm", {})
            cells.append(f"{w.get('mean', float('nan')):7.2f} (+{x.get('wrist_above_floor', float('nan')):.2f})")
        print(f"{m:24s} " + " ".join(f"{c:>16s}" for c in cells))
    print(f"SAVED {out}\nGAP EVAL OK")


if __name__ == "__main__":
    main()
