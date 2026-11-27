"""Frozen DINOv2 features for HOT3D Lance frames (same output format as encode.py).

Feature per frame (float16, 5*384 = 1920): [CLS, grid00, grid01, grid10, grid11], where grid = the
16x16 patch tokens (224 px / patch 14) averaged in a 2x2 grid (8x8 patches each), row-major over the
STORED (unrotated) image. Tokens are HF Dinov2Model last_hidden_state (after the final LayerNorm).
CLS-only = columns [:384].

  python hot3d_tools/dino_encode.py <lance_dir> <out_dir> --splits train,val --stride 10
      [--model facebook/dinov2-small]   (HF cache: $HF_HOME, set to scratch by the sbatch)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encode import Chunks, MEAN, STD  # noqa: E402


def load_dino(name):
    from transformers import Dinov2Model
    return Dinov2Model.from_pretrained(name).eval()


@torch.no_grad()
def feats(model, x):
    """x [B,3,224,224] normalized -> [B,1920]"""
    h = model(pixel_values=x).last_hidden_state                 # [B,1+256,384]
    cls, p = h[:, 0], h[:, 1:]
    B, N, D = p.shape
    g = int(round(N ** 0.5))
    p = p.reshape(B, 2, g // 2, 2, g // 2, D).mean((2, 4)).reshape(B, 4 * D)
    return torch.cat([cls, p], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lance_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--fov", type=int, default=110)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="facebook/dinov2-small")
    a = ap.parse_args()
    import lance
    assert "test" not in a.splits, "no test runs (user rule)"
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_dino(a.model).to(dev)
    info = dict(policy=f"dinov2:{a.model}", stride=a.stride, fov=a.fov, layout="CLS,g00,g01,g10,g11 x 384", splits={})
    for split in a.splits.split(","):
        path = f"{a.lance_dir}/hot3d_{split}_fov{a.fov}.lance"
        n = lance.dataset(path).count_rows()
        rows = np.arange(0, n, a.stride, dtype=np.int64)
        dl = torch.utils.data.DataLoader(Chunks(path, rows, a.chunk), batch_size=None, num_workers=a.workers,
                                         multiprocessing_context="spawn" if a.workers else None)
        out, t0 = [], time.time()
        for k, (r, x) in enumerate(dl):
            x = ((x.float() / 255 - MEAN) / STD).to(dev, non_blocking=True)
            amp = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(dev, dtype=amp, enabled=dev == "cuda"):
                f = feats(model, x)
            out.append(f.float().cpu().numpy().astype(np.float16))
            if k % 50 == 0:
                done = sum(len(e) for e in out)
                print(f"[{split}] {done}/{len(rows)} {done / max(time.time() - t0, 1e-6):.0f} fps", flush=True)
        E = np.concatenate(out)
        assert len(E) == len(rows) and np.isfinite(E).all()
        np.save(f"{a.out_dir}/emb_{split}.npy", E)
        np.save(f"{a.out_dir}/rows_{split}.npy", rows)
        info["splits"][split] = dict(n=len(rows), dim=int(E.shape[1]))
        print(f"SAVED {split} {E.shape} in {time.time() - t0:.0f}s", flush=True)
    json.dump(info, open(f"{a.out_dir}/info.json", "w"), indent=1)


if __name__ == "__main__":
    main()
