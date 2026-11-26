"""Raw fisheye RGB frames for the TimeWarp baseline (warp from the ORIGINAL fisheye
into the pinhole view, so edges are not black).

Save (hot3d env, PYTHONPATH=<hot3d repo>/hot3d; needs recording.vrs):
  python hot3d_tools/fisheye.py save <seq_folder> <out.npz> [--size 512] [--check <proc npz>]

Per recording, one uncompressed .npz:
  timestamps_ns [T]          same timeline as the pinhole npz / Lance rows
  jpeg, offsets [T+1]        frame i = jpeg[offsets[i]:offsets[i+1]] (JPEG, size x size RGB,
                             NOT rotated, area-downsampled from 1408x1408)
  proj_params [T,P], T_device_camera [T,3,4], valid_radius [T], max_solid_angle [T]
                             per-frame ONLINE calibration at FULL resolution (1408)
  model_type, label, serial, full_size, size

Use:
  f = FisheyeRecording(path); img = f.image(i); calib = f.calib(i)   # calib matches img (size x size)
"""
import argparse
import io
import math
import time

import numpy as np
from PIL import Image

RGB_ID = "214-1"


def _resize_offset(full, size):
    # rescale() requires new_res == (old_res - 2*offset) * scale, so no half-pixel centre fix:
    # leaves a ~0.44 px shift at 512 (~0.19 px at 224). The --check step measures the total error.
    return size / full, 0.0


class FisheyeRecording:
    def __init__(self, path):
        z = np.load(path)
        self.d = {k: z[k] for k in z.files}
        self.ts = self.d["timestamps_ns"]

    def __len__(self):
        return len(self.ts)

    def image(self, i):
        o = self.d["offsets"]
        return np.asarray(Image.open(io.BytesIO(self.d["jpeg"][o[i]:o[i + 1]].tobytes())).convert("RGB"))

    def calib(self, i):
        from projectaria_tools.core.calibration import CameraCalibration, CameraModelType
        from projectaria_tools.core.sophus import SE3
        d = self.d
        full, size = int(d["full_size"]), int(d["size"])
        T = np.eye(4)
        T[:3] = d["T_device_camera"][i]
        c = CameraCalibration(str(d["label"]), getattr(CameraModelType, str(d["model_type"])),
                              d["proj_params"][i].astype(np.float64), SE3.from_matrix(T), full, full,
                              float(d["valid_radius"][i]), float(d["max_solid_angle"][i]), str(d["serial"]))
        s, off = _resize_offset(full, size)
        return c.rescale(np.array([size, size], np.int32), s, np.array([off, off]))


def pinhole_from_fisheye(img, fish, fov, out_size=224, render=448):
    """Same undistortion as preprocess_sequence.py, but from a (downsampled) fisheye image."""
    from projectaria_tools.core.calibration import distort_by_calibration, get_linear_camera_calibration
    f_r = (render / 2) / math.tan(math.radians(fov / 2))
    pin = get_linear_camera_calibration(render, render, f_r, "camera-rgb", fish.get_transform_device_camera())
    und = distort_by_calibration(img, pin, fish)
    return np.asarray(Image.fromarray(und).resize((out_size, out_size), Image.BOX))


def save(a):
    from dataset_api import Hot3dDataProvider
    from projectaria_tools.core.stream_id import StreamId
    t0 = time.time()
    # Hot3dDataProvider requires an object library even though only images are read here
    from data_loaders.loader_object_library import load_object_library
    p = Hot3dDataProvider(sequence_folder=a.seq,
                          object_library=load_object_library(object_library_folderpath=a.assets),
                          mano_hand_model=None)
    dev = p.device_data_provider
    rgb = StreamId(RGB_ID)
    ts = np.array(dev.get_sequence_timestamps(rgb), dtype=np.int64)
    T = len(ts)
    blobs, params, Tdc, vr, msa = [], [], np.zeros((T, 3, 4)), np.zeros(T), np.zeros(T)
    meta = {}
    for i, t in enumerate(ts):
        t = int(t)
        img = dev.get_image(t, rgb)
        Tdc_i, fish = dev.get_online_camera_calibration(rgb, timestamp_ns=t)
        full = int(fish.get_image_size()[0])
        buf = io.BytesIO()
        Image.fromarray(img).resize((a.size, a.size), Image.BOX).save(buf, format="JPEG", quality=a.quality)
        blobs.append(np.frombuffer(buf.getvalue(), np.uint8))
        params.append(np.asarray(fish.projection_params(), np.float64))
        Tdc[i] = Tdc_i.to_matrix()[:3]
        r = fish.get_valid_radius()
        vr[i] = r if r is not None else np.nan
        msa[i] = fish.get_max_solid_angle()
        if not meta:
            meta = dict(model_type=str(fish.model_name()).split(".")[-1], label=fish.get_label(),
                        serial=fish.get_serial_number(), full_size=full)
        if i % 1000 == 0:
            print(f"{i}/{T} {time.time() - t0:.0f}s", flush=True)
    offsets = np.concatenate([[0], np.cumsum([len(b) for b in blobs])]).astype(np.int64)
    np.savez(a.out + ".tmp.npz", timestamps_ns=ts, jpeg=np.concatenate(blobs), offsets=offsets,
             proj_params=np.stack(params), T_device_camera=Tdc, valid_radius=vr, max_solid_angle=msa,
             size=np.int32(a.size), **meta)
    import os
    os.replace(a.out + ".tmp.npz", a.out)
    print(f"SAVED {a.out} T={T} {os.path.getsize(a.out) / 1e6:.0f} MB in {time.time() - t0:.0f}s", flush=True)

    if a.check:
        f = FisheyeRecording(a.out)
        z = np.load(a.check)
        assert (z["timestamps_ns"] == f.ts).all(), "timestamps differ from the pinhole npz"
        px, fov = z["pixels"], float(z["fov_deg"])
        errs = []
        for i in np.linspace(0, T - 1, 5).astype(int):
            rec = pinhole_from_fisheye(f.image(i), f.calib(i), fov, out_size=px.shape[1])
            errs.append(np.abs(rec.astype(np.float32) - px[i]).mean())
        print(f"CHECK pinhole rebuilt from saved fisheye vs stored frames: mean abs err "
              f"{np.round(errs, 2).tolist()} /255", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save")
    s.add_argument("seq")
    s.add_argument("assets")
    s.add_argument("out")
    s.add_argument("--size", type=int, default=512)
    s.add_argument("--quality", type=int, default=90)
    s.add_argument("--check", default="")
    a = ap.parse_args()
    save(a)
