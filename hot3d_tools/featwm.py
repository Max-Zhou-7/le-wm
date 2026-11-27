"""World model on FROZEN DINOv2 features (predictor only), with the same interface as jepa.JEPA
(encode / action_encoder / predict), so lewm_forecast.Forecaster, gap_eval and rq2 work unchanged.

Feature space = raw DINOv2 features of dino_encode.py (version cls: 384, cls_grid: 1920). Internally
the predictor works on standardized features ((f - mu) / sd, train-table stats); predict() takes and
returns RAW features. Predictor / action encoder / pred_proj = the LeWM modules (module.py), with
input_dim = feature dim and hidden 384.

Checkpoint folder: <dir>/featwm.pt (state dict incl. mu, sd) + <dir>/config.json.
Policy string for encode.load_model: "feat:<dir>" (absolute or relative to $STABLEWM_HOME/checkpoints).
"""
import json
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module import ARPredictor, Embedder, MLP  # noqa: E402

VERSIONS = {"cls": 384, "cls_grid": 1920}


class FeatWM(nn.Module):
    def __init__(self, version="cls_grid", act_dim=45, hidden=384, depth=6, heads=16, mlp_dim=2048, dim_head=64,
                 dropout=0.1, history=3, dino="facebook/dinov2-small"):
        super().__init__()
        D = VERSIONS[version]
        self.cfg = dict(version=version, act_dim=act_dim, hidden=hidden, depth=depth, heads=heads, mlp_dim=mlp_dim,
                        dim_head=dim_head, dropout=dropout, history=history, dino=dino)
        self.D = D
        self.register_buffer("mu", torch.zeros(D))
        self.register_buffer("sd", torch.ones(D))
        self.action_encoder = Embedder(input_dim=act_dim, emb_dim=D)
        self.predictor = ARPredictor(num_frames=history, depth=depth, heads=heads, mlp_dim=mlp_dim, input_dim=D,
                                     hidden_dim=hidden, output_dim=hidden, dim_head=dim_head, dropout=dropout)
        self.pred_proj = MLP(input_dim=hidden, output_dim=D, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
        self._dino = None

    # standardized space (training)
    def predict_std(self, e, act_emb):
        p = self.predictor(e, act_emb)
        B, T, H = p.shape
        return self.pred_proj(p.reshape(B * T, H)).reshape(B, T, -1)

    # raw feature space (inference, same as JEPA.predict)
    def predict(self, emb, act_emb):
        return self.predict_std((emb - self.mu) / self.sd, act_emb) * self.sd + self.mu

    def encode(self, info):
        """Frozen DINOv2 -> raw features of this version (for TimeWarp images in gap_eval)."""
        from dino_encode import feats, load_dino
        if self._dino is None:
            self._dino = [load_dino(self.cfg["dino"]).to(self.mu.device).eval()]
        px = info["pixels"].float()
        b, t = px.shape[:2]
        f = feats(self._dino[0], px.flatten(0, 1))[:, :self.D].float()
        info["emb"] = f.reshape(b, t, -1)
        return info

    def save(self, d):
        os.makedirs(d, exist_ok=True)
        torch.save(self.state_dict(), f"{d}/featwm.pt")
        json.dump(self.cfg, open(f"{d}/config.json", "w"), indent=1)


def load_featwm(d):
    if not os.path.isabs(d):
        d = os.path.join(os.environ.get("STABLEWM_HOME", ""), "checkpoints", d)
    cfg = json.load(open(f"{d}/config.json"))
    m = FeatWM(**cfg)
    m.load_state_dict(torch.load(f"{d}/featwm.pt", map_location="cpu"))
    return m.eval()
