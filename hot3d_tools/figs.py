"""Check figures (val only). Frames are rotated 90 deg clockwise for display only.

  python hot3d_tools/figs.py timewarp <lance_dir> <fisheye_dir> figs/timewarp_check_val.png
  python hot3d_tools/figs.py actions  <lance_dir> figs/actions_val.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_eval import FS, load_split, anchors  # noqa: E402
from actions import mat_to_rotvec  # noqa: E402

KS = [1, 4, 8]


def show(img):
    return np.rot90(img, k=-1)


def frames(D, rows):
    import torch
    from torchvision.io import decode_jpeg
    blobs = D["ds"].take([int(r) for r in rows], columns=["pixels"])["pixels"].to_pylist()
    return [decode_jpeg(torch.frombuffer(bytearray(b), dtype=torch.uint8)).permute(1, 2, 0).numpy() for b in blobs]


def fig_timewarp(a):
    from fisheye import FisheyeRecording
    from timewarp import Warper
    D = load_split(a.lance_dir, "val", 110)
    R = anchors(D, 15)
    # head rotation over the longest gap, per anchor; pick 3 recordings (different participants),
    # anchor with a clear but not extreme turn (closest to the 75th percentile within the recording)
    rot = np.array([np.degrees(np.linalg.norm(mat_to_rotvec(D["Rwc"][r].T @ D["Rwc"][r + FS * max(KS)]))) for r in R])
    recs = D["rec"][R]
    chosen, seen = [], set()
    for rec in sorted(set(recs), key=lambda x: -np.median(rot[recs == x])):
        p = rec.split("_")[0]
        if p in seen:
            continue
        m = np.flatnonzero(recs == rec)
        target = np.percentile(rot[m], 75)
        chosen.append(R[m[np.argmin(np.abs(rot[m] - target))]])
        seen.add(p)
        if len(chosen) == 3:
            break
    W = Warper()
    fig, ax = plt.subplots(len(chosen) * len(KS), 4, figsize=(10, 2.6 * len(chosen) * len(KS)))
    for c, r in enumerate(chosen):
        rec = D["rec"][r]
        f = FisheyeRecording(f"{a.fisheye_dir}/{rec}.npz")
        s = int(D["step"][r])
        img = f.image(s)
        for j, k in enumerate(KS):
            tg = r + FS * k
            Rsd = D["Rwc"][r].T @ D["Rwc"][tg]
            ang = np.degrees(np.linalg.norm(mat_to_rotvec(Rsd)))
            pred = W(img, f.d["proj_params"][s], int(f.d["full_size"]), Rsd, f.d["valid_radius"][s],
                     float(f.d["max_solid_angle"][s]))
            real_t, real_k = frames(D, [r, tg])
            diff = np.abs(pred.astype(np.float32) - real_k).mean(-1)
            diff0 = np.abs(real_t.astype(np.float32) - real_k).mean(-1)
            # score only pixels the warp covers (black = the old frame never saw that direction)
            cov = (pred.sum(-1) > 0) & (real_k.sum(-1) > 0)
            unc = 1 - cov.sum() / max((real_k.sum(-1) > 0).sum(), 1)
            diff_title = (f"|TW - real| {diff[cov].mean():.1f}/255 (repeat-last {diff0[cov].mean():.1f})\n"
                          f"on covered px; {100 * unc:.0f}% not covered")
            row = c * len(KS) + j
            for col, (im, title) in enumerate([
                    (real_t, f"{rec} real t" if j == 0 else "real t"),
                    (pred, f"TimeWarp t+{k} ({100 * k} ms, head turn {ang:.1f} deg)"),
                    (real_k, f"real t+{k}"),
                    (diff, diff_title)]):
                axx = ax[row, col]
                if im.ndim == 2:
                    axx.imshow(show(im), cmap="magma", vmin=0, vmax=80)
                else:
                    axx.imshow(show(im))
                axx.set_title(title, fontsize=7)
                axx.axis("off")
    fig.suptitle("TimeWarp check (val). Static background should line up; moving hands should not.\n"
                 "Frames rotated 90 deg cw for display; black corners = outside the lens limit.", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("SAVED", a.out)


def fig_actions(a):
    import lance
    ds = lance.dataset(f"{a.lance_dir}/hot3d_val_fov110.lance")
    tab = ds.to_table(columns=["episode_idx", "recording", "action", "t_sec"])
    ep = np.asarray(tab["episode_idx"].to_pylist()).reshape(-1)
    act = np.asarray(tab["action"].to_pylist(), np.float64)
    rec = np.asarray(tab["recording"].to_pylist())
    hz = 30.0
    head_rot = np.linalg.norm(act[:, 3:6], axis=1) * hz                 # deg/s
    lw = np.linalg.norm(act[:, 6:9], axis=1) * hz                        # cm/s
    rw = np.linalg.norm(act[:, 9:12], axis=1) * hz
    head_rot[act[:, 12] < 0.5] = np.nan
    lw[act[:, 13] < 0.5] = np.nan
    rw[act[:, 14] < 0.5] = np.nan
    # busiest 10 s (both hands moving) in one recording
    L = 300
    e0 = ep == ep[np.flatnonzero(rec == a.recording)[0]] if a.recording else ep == ep[0]
    idx = np.flatnonzero(e0)
    score = np.convolve(np.nan_to_num(lw[idx]) + np.nan_to_num(rw[idx]), np.ones(L), "valid")
    s0 = idx[int(np.argmax(score))]
    rows = np.arange(s0, s0 + L)
    t = (rows - s0) / hz
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(3, 10, height_ratios=[1.2, 1.2, 1.0])
    a1 = fig.add_subplot(gs[0, :])
    a1.plot(t, head_rot[rows], color="tab:purple", lw=1)
    a1.set_ylabel("head rotation\nspeed (deg/s)")
    a1.set_title(f"val {rec[s0]}, rows {s0 - idx[0]}-{s0 - idx[0] + L} (30 fps actions; gaps = invalid flag)", fontsize=9)
    a2 = fig.add_subplot(gs[1, :], sharex=a1)
    a2.plot(t, lw[rows], label="left wrist", color="tab:blue", lw=1)
    a2.plot(t, rw[rows], label="right wrist", color="tab:orange", lw=1)
    a2.set_ylabel("wrist speed\n(cm/s)")
    a2.set_xlabel("time (s)")
    a2.legend(fontsize=8, loc="upper right")
    for k in range(10):
        r = int(s0 + k * hz + hz / 2)
        a1.axvline((r - s0) / hz, color="gray", lw=0.5, ls=":")
        a2.axvline((r - s0) / hz, color="gray", lw=0.5, ls=":")
    import torch
    from torchvision.io import decode_jpeg
    blobs = ds.take([int(s0 + k * hz + hz / 2) for k in range(10)], columns=["pixels"])["pixels"].to_pylist()
    for k, b in enumerate(blobs):
        im = decode_jpeg(torch.frombuffer(bytearray(b), dtype=torch.uint8)).permute(1, 2, 0).numpy()
        ax = fig.add_subplot(gs[2, k])
        ax.imshow(show(im))
        ax.set_title(f"{k + 0.5:.1f} s", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("SAVED", a.out)


def fig_wrists(a):
    """Sanity check of probe targets: project the GT wrist (camera frame) with K into the stored frames."""
    import json
    D = load_split(a.lance_dir, "val", 110)
    K = np.asarray(json.load(open(f"{a.lance_dir}/hot3d_fov110_meta.json"))["K"], np.float64)
    from gap_eval import to_cam
    rng = np.random.default_rng(1)
    rows = np.sort(rng.choice(np.flatnonzero(D["hav"].all(1) & D["hv"]), 12, replace=False))
    imgs = frames(D, rows)
    fig, ax = plt.subplots(2, 6, figsize=(18, 6.5))
    for i, (r, im) in enumerate(zip(rows, imgs)):
        w = to_cam(D, np.array([r]), D["wrist_w"][[r]])[0] / 100.0     # [2,3] m
        axx = ax.flat[i]
        axx.imshow(im)                                                  # NOT rotated: K is for the stored frame
        for k, c in enumerate(["cyan", "orange"]):
            u = K[0, 0] * w[k, 0] / w[k, 2] + K[0, 2]
            v = K[1, 1] * w[k, 1] / w[k, 2] + K[1, 2]
            axx.plot(u, v, "o", ms=9, mfc="none", mec=c, mew=2)
            axx.text(u + 4, v, f"{'LR'[k]} z={100 * w[k, 2]:.0f}cm", color=c, fontsize=7)
        axx.set_title(f"{D['rec'][r]} row {r}", fontsize=7)
        axx.axis("off")
    fig.suptitle("GT wrist (cyan = left, orange = right) projected with K into the stored frame (NOT rotated). "
                 "Circles should sit on the wrists.", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=100)
    print("K", K.tolist())
    print("SAVED", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["timewarp", "actions", "wrists"])
    ap.add_argument("lance_dir")
    ap.add_argument("rest", nargs="+")
    ap.add_argument("--recording", default="")
    a = ap.parse_args()
    if a.cmd == "timewarp":
        a.fisheye_dir, a.out = a.rest
        fig_timewarp(a)
    elif a.cmd == "wrists":
        (a.out,) = a.rest
        fig_wrists(a)
    else:
        (a.out,) = a.rest
        fig_actions(a)

