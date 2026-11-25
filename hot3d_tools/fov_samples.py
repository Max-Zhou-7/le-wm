"""Contact sheet to choose the pinhole field of view for HOT3D Aria RGB frames.

Rows: frames where a hand is near the fisheye border (+ one ordinary frame).
Columns: raw fisheye, then pinhole undistortions at several horizontal FOVs.
Dots: GT wrist positions projected into each image (red = left, green = right),
so this also checks pose-to-frame alignment.
Frames are shown rotated 90 deg clockwise (upright) for viewing only.

Usage (hot3d env, PYTHONPATH=<hot3d repo>/hot3d):
  python hot3d_tools/fov_samples.py <seq_folder> <assets_folder> <out.png>
"""
import csv
import math
import sys

import numpy as np
from PIL import Image, ImageDraw
from dataset_api import Hot3dDataProvider
from data_loaders.loader_object_library import load_object_library
from data_loaders.loader_hand_poses import Handedness
from projectaria_tools.core.calibration import (
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

FOVS = [90, 100, 110, 120]  # horizontal = vertical FOV of the square pinhole
SIZE = 448                  # undistort at 2x, stored frames will be 224
seq, assets, out = sys.argv[1:4]
RGB = StreamId("214-1")

p = Hot3dDataProvider(
    sequence_folder=seq,
    object_library=load_object_library(object_library_folderpath=assets),
    mano_hand_model=None,
)
dev = p.device_data_provider
ts_all = np.array(dev.get_sequence_timestamps())

# pick frames: hand box centre closest to the fisheye border (W=H=1408)
rows = [r for r in csv.DictReader(open(f"{seq}/box2d_hands.csv")) if r["stream_id"] == "214-1"]
def border_dist(r):
    x = (float(r["x_min[pixel]"]) + float(r["x_max[pixel]"])) / 2
    y = (float(r["y_min[pixel]"]) + float(r["y_max[pixel]"])) / 2
    return math.hypot(x - 704, y - 704)
rows = [r for r in rows if r["x_min[pixel]"] not in ("", "nan") and float(r["visibility_ratio[%]"] or 0) > 0.3]
rows.sort(key=border_dist, reverse=True)
picked = []
for r in rows:
    t = int(r["timestamp[ns]"])
    if all(abs(t - q) > 5e9 for q in picked):  # at least 5 s apart
        picked.append(t)
    if len(picked) == 3:
        break
picked.append(int(ts_all[len(ts_all) // 2]))

q = dict(time_query_options=TimeQueryOptions.CLOSEST, time_domain=TimeDomain.TIME_CODE)
tiles = []
for t in picked:
    img = dev.get_image(t, RGB)
    T_dev_cam, fish = dev.get_online_camera_calibration(RGB, timestamp_ns=t)
    T_w_dev = p.device_pose_data_provider.get_pose_at_timestamp(timestamp_ns=t, **q).pose3d.T_world_device
    T_cam_w = (T_w_dev @ T_dev_cam).inverse()
    hands = p.umetrack_hand_data_provider.get_pose_at_timestamp(timestamp_ns=t, **q).pose3d_collection.poses
    pts = {h: (T_cam_w @ v.wrist_pose).translation()[0] for h, v in hands.items() if v.wrist_pose is not None}

    def draw(im, calib, scale):
        im = Image.fromarray(im).convert("RGB")
        d = ImageDraw.Draw(im)
        for h, pc in pts.items():
            uv = calib.project(pc)
            if uv is not None:
                u, v = uv * scale
                c = (255, 0, 0) if h == Handedness.Left else (0, 255, 0)
                d.ellipse([u - 6, v - 6, u + 6, v + 6], outline=c, width=3)
        return im

    row = [draw(img, fish, 1.0).resize((SIZE, SIZE)).rotate(-90)]
    for fov in FOVS:
        f = (SIZE / 2) / math.tan(math.radians(fov / 2))
        pin = get_linear_camera_calibration(SIZE, SIZE, f, "camera-rgb", T_dev_cam)
        und = distort_by_calibration(img, pin, fish)
        row.append(draw(und, pin, 1.0).rotate(-90))
    tiles.append(row)

W = SIZE * len(tiles[0])
sheet = Image.new("RGB", (W, SIZE * len(tiles) + 30), "white")
d = ImageDraw.Draw(sheet)
for j, lab in enumerate(["raw fisheye"] + [f"pinhole {f} deg" for f in FOVS]):
    d.text((j * SIZE + 10, 8), lab, fill=(0, 0, 0))
for i, row in enumerate(tiles):
    for j, im in enumerate(row):
        sheet.paste(im, (j * SIZE, 30 + i * SIZE))
sheet.save(out)
print("saved", out, "frames", picked)
