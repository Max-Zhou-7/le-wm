"""Print the structure of one HOT3D Aria sequence (streams, rates, poses).

Run in the hot3d env from the HOT3D repo's hot3d/ folder on PYTHONPATH:
  PYTHONPATH=~/ondemand/dev/LeWM_forecasting/hot3d/hot3d \
  python hot3d_tools/inspect_sequence.py /scratch/$USER/hot3d/aria/P0003_c701bd11 /scratch/$USER/hot3d/assets/assets
"""
import sys

import numpy as np
from dataset_api import Hot3dDataProvider
from data_loaders.loader_object_library import load_object_library
from projectaria_tools.core.stream_id import StreamId

seq, assets = sys.argv[1], sys.argv[2]
p = Hot3dDataProvider(
    sequence_folder=seq,
    object_library=load_object_library(object_library_folderpath=assets),
    mano_hand_model=None,
)
print("metadata:", p.get_sequence_metadata())

dev = p.device_data_provider
streams = dev.get_image_stream_ids()
print("image streams:", [(str(s), dev.get_image_stream_label(s)) for s in streams])
for s in streams:
    ts = np.array(dev.get_sequence_timestamps())  # GT timeline
    try:
        ts = np.array(dev.get_image_stream_timestamps(s)) if hasattr(dev, "get_image_stream_timestamps") else ts
    except Exception:
        pass
    img = dev.get_image(int(ts[len(ts) // 2]), s)
    d = np.diff(ts) / 1e6
    print(f"{s} {dev.get_image_stream_label(s)}: n={len(ts)} dur={(ts[-1]-ts[0])/1e9:.1f}s "
          f"dt median={np.median(d):.2f}ms min={d.min():.2f} max={d.max():.2f} "
          f"img shape={None if img is None else img.shape} dtype={None if img is None else img.dtype}")

ts = np.array(dev.get_sequence_timestamps())
d = np.diff(ts) / 1e6
print(f"GT timeline: n={len(ts)} dt median={np.median(d):.2f}ms  (unique dts: {np.unique(np.round(d,1))[:10]})")

rgb = StreamId("214-1")
print("rgb camera calib:", dev.get_camera_calibration(rgb)[1] if isinstance(dev.get_camera_calibration(rgb), tuple) else dev.get_camera_calibration(rgb))

t = int(ts[len(ts) // 2])
hp = p.headset_pose_data_provider.get_pose_at_timestamp(t, None, 0, None) if False else None
for name in ["headset_pose_data_provider", "object_pose_data_provider", "hand_data_provider", "mano_hand_data_provider", "object_box2d_data_provider", "hand_box2d_data_provider", "object_library"]:
    print(name, "->", type(getattr(p, name, None)).__name__)

# poses at one timestamp
from data_loaders.loader_hand_poses import HandType  # noqa: F401
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
hdp = p.headset_pose_data_provider.get_pose_at_timestamp(
    timestamp_ns=t, time_query_options=TimeQueryOptions.CLOSEST, time_domain=TimeDomain.TIME_CODE)
print("headset pose @t:", hdp)
odp = p.object_pose_data_provider.get_pose_at_timestamp(
    timestamp_ns=t, time_query_options=TimeQueryOptions.CLOSEST, time_domain=TimeDomain.TIME_CODE)
print("object poses @t:", {k: v.T_world_object.translation() for k, v in odp.pose3d_collection.poses.items()})
hp = p.hand_data_provider.get_pose_at_timestamp(
    timestamp_ns=t, time_query_options=TimeQueryOptions.CLOSEST, time_domain=TimeDomain.TIME_CODE)
print("hand poses @t:", {str(k): v.wrist_pose.translation() for k, v in hp.pose3d_collection.poses.items()})
