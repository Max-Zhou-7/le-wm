import json, os, re, torch
from pathlib import Path
from hydra.utils import instantiate

home = Path(os.environ["STABLEWM_HOME"])
src, out = home / "hf_pusht", home / "pusht" / "lewm_object.ckpt"

def rename(k):
    k = re.sub(r"^encoder\.encoder\.layer\.(\d+)\.", r"encoder.layers.\1.", k)
    k = k.replace(".attention.attention.query", ".attention.q_proj")
    k = k.replace(".attention.attention.key", ".attention.k_proj")
    k = k.replace(".attention.attention.value", ".attention.v_proj")
    k = k.replace(".attention.output.dense", ".attention.o_proj")
    k = k.replace(".intermediate.dense", ".mlp.fc1")
    k = k.replace(".output.dense", ".mlp.fc2")
    return k

cfg = json.loads((src / "config.json").read_text())
model = instantiate(cfg)
old = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
sd = {rename(k): v for k, v in old.items() if not k.startswith("encoder.pooler")}

missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing:", missing)
print("unexpected:", unexpected)
assert not missing and not unexpected, "weights still do not match"

out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
print("SAVED", out, type(model).__name__)

# eval.py loads models with swm.wm.utils.load_pretrained, which wants a folder
# under $STABLEWM_HOME/checkpoints/ holding one .pt state dict + config.json.
folder = home / "checkpoints" / "pusht_released"
folder.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), folder / "weights.pt")
(folder / "config.json").write_text(json.dumps(cfg, indent=2))
print("SAVED", folder, "-> use policy=pusht_released")
