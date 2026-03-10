import numpy as np
import torch
from torchvision import transforms


def random_resized_crop(
    size: int = 224,
    scale=(0.7, 1.0),
    ratio=(3 / 4, 4 / 3),
) -> transforms.RandomResizedCrop:
    return transforms.RandomResizedCrop(size=size, scale=scale, ratio=ratio)


def get_phase4_ops(
    size: int = 224,
    scale=(0.7, 1.0),
    ratio=(3 / 4, 4 / 3),
):
    return [
        random_resized_crop(size=size, scale=scale, ratio=ratio)
    ]


def append_phase4_ops(base_ops: list, **kwargs) -> list:
    return list(base_ops) + get_phase4_ops(**kwargs)


def build_phase4_transform(base_ops: list, **kwargs) -> transforms.Compose:
    return transforms.Compose(append_phase4_ops(base_ops, **kwargs))


def _rand_bbox(size, lam):
    _, _, h, w = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    return x1, y1, x2, y2


def cutmix_data(x, y, alpha: float = 1.0):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    y_a = y
    y_b = y[index]

    x1, y1, x2, y2 = _rand_bbox(x.size(), lam)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam


def cutmix_loss(criterion, preds, y_a, y_b, lam: float):
    return lam * criterion(preds, y_a) + (1.0 - lam) * criterion(preds, y_b)