"""Preprocess one HOT3D Aria sequence into a compact .npz (all frames, 30 fps).

Frames: RGB stream 214-1, undistorted from Fisheye624 to a square pinhole camera
with the given FOV (rendered at --render px, then area-downsampled to --size px).
Frames are NOT rotated (Aria RGB is stored sideways; rotate 90 deg cw only for figures).

Poses are raw, aligned to each RGB timestamp (TIME_CODE, closest GT sample):
  T_world_device [T,7], T_device_camera [T,7] (online calibration),
  hand_wrist [T,2,7] (0=left,1=right, world frame), hand_joints [T,2,J] (UmeTrack),
  obj_pose [T,O,7] (world frame, O = all objects in the library, by BOP id).
  7-vectors are [tx,ty,tz,qw,qx,qy,qz] (metres), same order as the HOT3D CSVs.
Missing values are NaN; *_valid arrays say what is present; *_dt_ns is the
time gap between the frame and the pose sample used.
K is the pinhole intrinsics at --size px.

Usage (hot3d env, PYTHONPATH=<hot3d repo>/hot3d):
  python hot3d_tools/preprocess_sequence.py <seq_folder> <assets_folder> <out.npz> [--fov 110]
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
from PIL import Image
from dataset_api import Hot3dDataProvider
from data_loaders.loader_hand_poses import Handedness
from data_loaders.loader_object_library import load_object_library
from projectaria_tools.core.calibration import distort_by_calibration, get_linear_camera_calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

RGB = StreamId("214-1")
Q = dict(time_query_options=TimeQueryOptions.CLOSEST, time_domain=TimeDomain.TIME_CODE)


def se3_to_vec(T):
    """sophus SE3 -> [tx,ty,tz,qw,qx,qy,qz]."""
    t = np.asarray(T.translation()).reshape(-1)[:3]
    q = np.asarray(T.rotation().to_quat()).reshape(-1)[:4]  # [w,x,y,z]
    return np.concatenate([t, q])


def load_mask(seq, name, ts):
    path = f"{seq}/masks/mask_{name}.csv"
    if not os.path.exists(path):
        return np.full(len(ts), False)
    m = {int(r["timestamp[ns]"]): r["mask"] == "True"
         for r in csv.DictReader(open(path)) if r["stream_id"] == "214-1"}
    return np.array([m.get(int(t), False) for t in ts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("assets")
    ap.add_argument("out")
    ap.add_argument("--fov", type=float, default=110.0)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--render", type=int, default=448)
    ap.add_argument("--max-frames", type=int, default=0, help="debug: only the first N frames")
    a = ap.parse_args()
    t0 = time.time()

    meta = json.load(open(f"{a.seq}/metadata.json"))
    have_gt = bool(meta.get("have_hand_object_pose_gt", False))
    lib = json.load(open(f"{a.assets}/instance.json"))
    objs = sorted((v for v in lib.values() if v.get("instance_type") == "object"),
                  key=lambda v: int(v["instance_bop_id"]))
    obj_uids = [o["instance_id"] for o in objs]
    obj_index = {u: i for i, u in enumerate(obj_uids)}

    p = Hot3dDataProvider(sequence_folder=a.seq,
                          object_library=load_object_library(object_library_folderpath=a.assets),
                          mano_hand_model=None)
    dev = p.device_data_provider
    ts = np.array(dev.get_sequence_timestamps(RGB), dtype=np.int64)
    if a.max_frames:
        ts = ts[: a.max_frames]
    T = len(ts)

    f_r = (a.render / 2) / math.tan(math.radians(a.fov / 2))
    f_s = (a.size / 2) / math.tan(math.radians(a.fov / 2))
    K = np.array([[f_s, 0, (a.size - 1) / 2], [0, f_s, (a.size - 1) / 2], [0, 0, 1]], np.float32)

    pixels = np.zeros((T, a.size, a.size, 3), np.uint8)
    nan7 = lambda *s: np.full((*s, 7), np.nan, np.float32)
    T_world_device, T_device_camera = nan7(T), nan7(T)
    head_dt = np.full(T, np.iinfo(np.int64).max, np.int64)
    hand_wrist, hand_dt = nan7(T, 2), np.full(T, np.iinfo(np.int64).max, np.int64)
    hand_joints = None
    obj_pose, obj_dt = nan7(T, len(obj_uids)), np.full(T, np.iinfo(np.int64).max, np.int64)

    hands_p = p.umetrack_hand_data_provider if have_gt else None
    for i, t in enumerate(ts):
        t = int(t)
        img = dev.get_image(t, RGB)
        Tdc, fish = dev.get_online_camera_calibration(RGB, timestamp_ns=t)
        pin = get_linear_camera_calibration(a.render, a.render, f_r, "camera-rgb", Tdc)
        und = distort_by_calibration(img, pin, fish)
        pixels[i] = np.asarray(Image.fromarray(und).resize((a.size, a.size), Image.BOX))
        T_device_camera[i] = se3_to_vec(Tdc)

        hp = p.device_pose_data_provider.get_pose_at_timestamp(timestamp_ns=t, **Q)
        if hp is not None:
            T_world_device[i] = se3_to_vec(hp.pose3d.T_world_device)
            head_dt[i] = hp.time_delta_ns
        if hands_p is not None:
            h = hands_p.get_pose_at_timestamp(timestamp_ns=t, **Q)
            if h is not None:
                hand_dt[i] = h.time_delta_ns
                for hand, v in h.pose3d_collection.poses.items():
                    k = 0 if hand == Handedness.Left else 1
                    if v.wrist_pose is not None:
                        hand_wrist[i, k] = se3_to_vec(v.wrist_pose)
                    if v.joint_angles is not None:
                        if hand_joints is None:
                            hand_joints = np.full((T, 2, len(v.joint_angles)), np.nan, np.float32)
                        hand_joints[i, k] = v.joint_angles
        if have_gt:
            o = p.object_pose_data_provider.get_pose_at_timestamp(timestamp_ns=t, **Q)
            if o is not None:
                obj_dt[i] = o.time_delta_ns
                for uid, v in o.pose3d_collection.poses.items():
                    obj_pose[i, obj_index[uid]] = se3_to_vec(v.T_world_object)
        if i % 500 == 0:
            print(f"{i}/{T} {time.time() - t0:.0f}s", flush=True)

    if hand_joints is None:
        hand_joints = np.full((T, 2, 22), np.nan, np.float32)
    # CLOSEST lookup can reach across gaps in the GT (pilot: hand poses 100 ms away).
    # Poses more than MAX_DT_NS from the frame are treated as missing; *_dt_ns keeps the raw gap.
    MAX_DT_NS = 5_000_000
    T_world_device[np.abs(head_dt) > MAX_DT_NS] = np.nan
    far = np.abs(hand_dt) > MAX_DT_NS
    hand_wrist[far], hand_joints[far] = np.nan, np.nan
    obj_pose[np.abs(obj_dt) > MAX_DT_NS] = np.nan
    vis = {}
    for r in csv.DictReader(open(f"{a.seq}/box2d_hands.csv")) if os.path.exists(f"{a.seq}/box2d_hands.csv") else []:
        if r["stream_id"] == "214-1":
            vis[(int(r["timestamp[ns]"]), int(r["hand_index"]))] = float(r["visibility_ratio[%]"] or 0)
    hand_vis_ratio = np.array([[vis.get((int(t), k), np.nan) for k in (0, 1)] for t in ts], np.float32)

    np.savez_compressed(
        a.out,
        timestamps_ns=ts, pixels=pixels, K=K, fov_deg=np.float32(a.fov),
        T_world_device=T_world_device, head_dt_ns=head_dt, T_device_camera=T_device_camera,
        hand_wrist=hand_wrist, hand_joints=hand_joints, hand_dt_ns=hand_dt,
        hand_valid=~np.isnan(hand_wrist[..., 0]), hand_vis_ratio=hand_vis_ratio,
        obj_pose=obj_pose, obj_dt_ns=obj_dt, obj_valid=~np.isnan(obj_pose[..., 0]),
        obj_uids=np.array(obj_uids),
        qa_pass=load_mask(a.seq, "qa_pass", ts),
        mask_hand_visible=load_mask(a.seq, "hand_visible", ts),
        mask_object_visible=load_mask(a.seq, "object_visible", ts),
        mask_good_exposure=load_mask(a.seq, "good_exposure", ts),
        recording=meta["recording_name"], participant=meta["participant_id"], have_gt=have_gt,
    )
    print(f"SAVED {a.out} T={T} {os.path.getsize(a.out) / 1e6:.0f} MB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
