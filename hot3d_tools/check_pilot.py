"""Checks for preprocessed HOT3D sequences (pilot).

For each <proc_dir>/<seq>.npz:
  - shapes, valid fractions, pose-to-frame time gaps
  - stored head pose vs headset_trajectory.csv (quaternion order/sign check)
  - % of frames with each wrist inside the image at several pinhole FOVs
    ("in square" = projects into the square image; "in lens" = also inside the
    fisheye's valid area, i.e. real pixels rather than black corners)
  - mask fractions
Writes an overlay figure: stored frames (rotated upright only for viewing) with
projected wrists (red=left, green=right) and object centres (yellow), on random
frames and on the frames with the fastest wrist motion.

Usage (hot3d env, PYTHONPATH=<hot3d repo>/hot3d; needs recording.vrs for the lens check):
  python hot3d_tools/check_pilot.py <proc_dir> <aria_root> <out_prefix>
"""
import csv
import glob
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

proc_dir, aria_root, out_prefix = sys.argv[1:4]
FOVS = [100, 110]


def quat_to_R(q):  # q = [w,x,y,z]
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def to_mat(v):
    M = np.eye(4)
    M[:3, :3] = quat_to_R(v[3:7])
    M[:3, 3] = v[:3]
    return M


def lens_checker(seq_dir):
    """Returns f(p_cam) -> True if the fisheye RGB camera sees p_cam (valid area)."""
    try:
        from projectaria_tools.core import data_provider
        prov = data_provider.create_vrs_data_provider(f"{seq_dir}/recording.vrs")
        fish = prov.get_device_calibration().get_camera_calib("camera-rgb")
        return lambda pc: pc[2] > 0 and fish.project(pc) is not None
    except Exception as e:  # VRS deleted or unreadable
        print("  (no lens check:", e, ")")
        return None


figs = []
rows = []
for path in sorted(glob.glob(f"{proc_dir}/*.npz")):
    d = np.load(path)
    seq = str(d["recording"])
    T = len(d["timestamps_ns"])
    S = d["pixels"].shape[1]
    print(f"\n== {seq} participant={d['participant']} have_gt={bool(d['have_gt'])} T={T} "
          f"pixels={d['pixels'].shape} fov={float(d['fov_deg'])}")
    dts = np.diff(d["timestamps_ns"]) / 1e6
    print(f"  frame dt ms: median {np.median(dts):.2f} min {dts.min():.2f} max {dts.max():.2f}")
    for k in ["head", "hand", "obj"]:
        dt = d[f"{k}_dt_ns"].astype(np.float64)
        ok = dt < np.iinfo(np.int64).max
        dt = np.abs(dt[ok]) / 1e6
        if len(dt):
            print(f"  {k} pose found {ok.mean():.3f} of frames; |dt| ms median {np.median(dt):.3f} "
                  f"p99 {np.percentile(dt, 99):.3f} max {dt.max():.3f}")
    print(f"  head valid {(~np.isnan(d['T_world_device'][:, 0])).mean():.3f}  "
          f"hand valid L/R {d['hand_valid'].mean(0).round(3)}  "
          f"objects present in seq: {int(d['obj_valid'].any(0).sum())}, "
          f"obj valid (of present) {d['obj_valid'][:, d['obj_valid'].any(0)].mean():.3f}")
    print(f"  masks: qa_pass {d['qa_pass'].mean():.3f} hand_visible {d['mask_hand_visible'].mean():.3f} "
          f"object_visible {d['mask_object_visible'].mean():.3f} good_exposure {d['mask_good_exposure'].mean():.3f}; "
          f"box2d hand vis>0 L/R {np.nan_to_num(d['hand_vis_ratio'] > 0).mean(0).round(3)}")

    # head pose vs csv
    csvp = f"{aria_root}/{seq}/headset_trajectory.csv"
    if os.path.exists(csvp):
        ref = {int(r["timestamp[ns]"]): np.array([float(r[k]) for k in list(r)[2:9]])
               for r in csv.DictReader(open(csvp))}
        err_t, err_q = [], []
        for i in range(0, T, 50):
            t = int(d["timestamps_ns"][i])
            if t in ref and not np.isnan(d["T_world_device"][i, 0]):
                v, r = d["T_world_device"][i], ref[t]
                err_t.append(np.abs(v[:3] - r[:3]).max())
                err_q.append(min(np.abs(v[3:] - r[3:]).max(), np.abs(v[3:] + r[3:]).max()))
        if err_t:
            print(f"  head vs csv ({len(err_t)} exact-timestamp frames): max |dt| {max(err_t):.2e} m, "
                  f"max |dq| {max(err_q):.2e}")

    # wrist in image
    lens = lens_checker(f"{aria_root}/{seq}")
    res = {}
    pc_all = np.full((T, 2, 3), np.nan)
    for i in range(T):
        if np.isnan(d["T_world_device"][i, 0]) or np.isnan(d["T_device_camera"][i, 0]):
            continue
        T_cw = np.linalg.inv(to_mat(d["T_world_device"][i]) @ to_mat(d["T_device_camera"][i]))
        for k in range(2):
            if d["hand_valid"][i, k]:
                pc_all[i, k] = (T_cw @ np.append(d["hand_wrist"][i, k, :3], 1))[:3]
    have = ~np.isnan(pc_all[..., 0])
    in_lens = np.zeros((T, 2), bool)
    if lens is not None:
        for i, k in zip(*np.nonzero(have)):
            in_lens[i, k] = lens(pc_all[i, k])
    for fov in FOVS:
        f = (S / 2) / math.tan(math.radians(fov / 2))
        z = pc_all[..., 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            u = f * pc_all[..., 0] / z + (S - 1) / 2
            v = f * pc_all[..., 1] / z + (S - 1) / 2
            sq = have & (z > 0) & (u >= 0) & (u < S) & (v >= 0) & (v < S)
        both = sq & in_lens if lens is not None else sq
        res[fov] = (sq.sum(0), both.sum(0))
    n = have.sum(0)
    for fov in FOVS:
        sq, both = res[fov]
        print(f"  wrist in image @ {fov} deg: in square L {100 * sq[0] / n[0]:.1f}% R {100 * sq[1] / n[1]:.1f}%"
              + (f" | in square & lens L {100 * both[0] / n[0]:.1f}% R {100 * both[1] / n[1]:.1f}%" if lens else ""))
    rows.append((seq, n, res))

    # overlay figure
    K = d["K"]
    speed = np.nan_to_num(np.linalg.norm(np.diff(d["hand_wrist"][..., :3], axis=0), axis=-1).max(-1))
    rng = np.random.default_rng(0)
    pick = list(np.argsort(speed)[-2:] + 1) + list(rng.choice(T, 2, replace=False))
    for i in pick:
        im = Image.fromarray(d["pixels"][i]).resize((2 * S, 2 * S))
        dr = ImageDraw.Draw(im)
        T_cw = np.linalg.inv(to_mat(d["T_world_device"][i]) @ to_mat(d["T_device_camera"][i]))
        pts = [(d["hand_wrist"][i, k, :3], (255, 0, 0) if k == 0 else (0, 255, 0)) for k in range(2) if d["hand_valid"][i, k]]
        pts += [(d["obj_pose"][i, j, :3], (255, 220, 0)) for j in np.nonzero(d["obj_valid"][i])[0]]
        for pw, c in pts:
            pc = (T_cw @ np.append(pw, 1))[:3]
            if pc[2] > 0:
                uv = (K @ pc)[:2] / pc[2] * 2
                dr.ellipse([uv[0] - 5, uv[1] - 5, uv[0] + 5, uv[1] + 5], outline=c, width=3)
        im = im.rotate(-90)
        ImageDraw.Draw(im).text((5, 5), f"{seq} f{i}" + (" fast" if i in pick[:2] else ""), fill=(255, 255, 255))
        figs.append(im)

print("\n== ALL SEQUENCES: wrist-in-image (frames with a wrist pose)")
for fov in FOVS:
    n = sum(r[1] for r in rows)
    sq = sum(r[2][fov][0] for r in rows)
    both = sum(r[2][fov][1] for r in rows)
    print(f"  {fov} deg: in square L {100 * sq[0] / n[0]:.1f}% R {100 * sq[1] / n[1]:.1f}% | "
          f"in square & lens L {100 * both[0] / n[0]:.1f}% R {100 * both[1] / n[1]:.1f}%")

if figs:
    W = figs[0].width
    cols = 4
    sheet = Image.new("RGB", (W * cols, W * math.ceil(len(figs) / cols)))
    for j, im in enumerate(figs):
        sheet.paste(im, ((j % cols) * W, (j // cols) * W))
    sheet.save(f"{out_prefix}_overlay.png")
    print("saved", f"{out_prefix}_overlay.png")
