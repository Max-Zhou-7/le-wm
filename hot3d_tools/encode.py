"""Encode HOT3D Lance frames with a frozen LeWM encoder (+ projector), one embedding per row.

Output: <out_dir>/emb_<split>.npy  float16 [N, D] for rows 0, stride, 2*stride, ...
        <out_dir>/rows_<split>.npy int64 [N]     table row index of each embedding
        <out_dir>/info.json                       policy, stride, dims
The embedding is model.encode(...)["emb"] (projector output, what the predictor sees).

Usage (lewm env; GPU or CPU):
  python hot3d_tools/encode.py <policy> <lance_dir> <out_dir> [--splits val,test] [--stride 1]
  <policy>: as in eval.py (relative to $STABLEWM_HOME/checkpoints), or "random:<policy>"
            for the same architecture with untrained weights (floor for the probes).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # jepa.py, module.py

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Chunks(torch.utils.data.Dataset):
    def __init__(self, path, rows, chunk):
        self.path, self.rows, self.chunk = path, rows, chunk
        self.ds = None

    def __len__(self):
        return (len(self.rows) + self.chunk - 1) // self.chunk

    def __getitem__(self, i):
        import lance
        from torchvision.io import decode_jpeg, ImageReadMode
        if self.ds is None:
            self.ds = lance.dataset(self.path)
        r = self.rows[i * self.chunk:(i + 1) * self.chunk]
        blobs = self.ds.take(r.tolist(), columns=["pixels"])["pixels"].to_pylist()
        imgs = decode_jpeg([torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in blobs],
                           mode=ImageReadMode.RGB)
        return torch.from_numpy(r), torch.stack(imgs)


def load_model(policy):
    if policy.startswith("feat:"):                     # predictor on frozen DINOv2 features (featwm.py)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from featwm import load_featwm
        return load_featwm(policy.split(":", 1)[1])
    import stable_worldmodel as swm
    rand = policy.startswith("random:")
    model = swm.wm.utils.load_pretrained(policy.split(":", 1)[1] if rand else policy)
    if rand:
        torch.manual_seed(0)
        for m in model.modules():
            if hasattr(m, "reset_parameters") and m is not model:
                m.reset_parameters()
    return model.eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("lance_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--fov", type=int, default=110)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import lance
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.policy).to(dev)
    info = dict(policy=a.policy, stride=a.stride, fov=a.fov, splits={})
    for split in a.splits.split(","):
        path = f"{a.lance_dir}/hot3d_{split}_fov{a.fov}.lance"
        n = lance.dataset(path).count_rows()
        rows = np.arange(0, n, a.stride, dtype=np.int64)
        dl = torch.utils.data.DataLoader(Chunks(path, rows, a.chunk), batch_size=None, num_workers=a.workers,
                                         multiprocessing_context="spawn" if a.workers else None)
        embs, t0 = [], time.time()
        for k, (r, x) in enumerate(dl):
            x = ((x.float() / 255 - MEAN) / STD).to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                e = model.encode({"pixels": x[:, None]})["emb"][:, 0]
            embs.append(e.float().cpu().numpy().astype(np.float16))
            if k % 50 == 0:
                done = sum(len(e) for e in embs)
                print(f"[{split}] {done}/{len(rows)} {done / max(time.time() - t0, 1e-6):.0f} fps", flush=True)
        E = np.concatenate(embs)
        assert len(E) == len(rows) and np.isfinite(E).all()
        np.save(f"{a.out_dir}/emb_{split}.npy", E)
        np.save(f"{a.out_dir}/rows_{split}.npy", rows)
        info["splits"][split] = dict(n=len(rows), dim=int(E.shape[1]))
        print(f"SAVED {split} {E.shape} in {time.time() - t0:.0f}s", flush=True)
    json.dump(info, open(f"{a.out_dir}/info.json", "w"), indent=1)


if __name__ == "__main__":
    main()
