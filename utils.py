import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class Hot3dActionTransform:
    """HOT3D action (T, frameskip*15): per raw step [head t(3), head r(3), L(3), R(3), flags(3)].
    Z-scores dims 0:12 with stats from VALID rows only, leaves the flags as 0/1, puts invalid
    parts back to exactly 0, then applies the variant:
      none        all zeros (the predictor gets no motion input)
      head        head part + head flag only
      head_hands  everything"""

    def __init__(self, mean, std, variant):
        assert variant in ("none", "head", "head_hands"), variant
        self.mean, self.std, self.variant = mean, std, variant

    def __call__(self, x):
        T = x.shape[0]
        a = torch.nan_to_num(x.float(), 0.0).reshape(T, -1, 15).clone()
        flags = a[..., 12:15]
        a[..., :12] = (a[..., :12] - self.mean) / self.std
        a[..., 0:6] *= flags[..., 0:1]
        a[..., 6:9] *= flags[..., 1:2]
        a[..., 9:12] *= flags[..., 2:3]
        if self.variant == "none":
            a.zero_()
        elif self.variant == "head":
            a[..., 6:12] = 0
            a[..., 13:15] = 0
        return a.reshape(T, -1)


def get_hot3d_action_transform(dataset, variant):
    act = torch.from_numpy(np.asarray(dataset.get_col_data("action"), dtype=np.float32))
    mean, std = torch.zeros(12), torch.ones(12)
    for part, (lo, hi) in enumerate([(0, 6), (6, 9), (9, 12)]):
        v = act[act[:, 12 + part] > 0.5, lo:hi]
        mean[lo:hi], std[lo:hi] = v.mean(0), v.std(0).clamp_min(1e-6)
    return dt.transforms.WrapTorchTransform(Hot3dActionTransform(mean, std, variant),
                                            source="action", target="action")


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )