from typing import List

from torchvision import transforms


def jitter_brightness(brightness: float = 0.2) -> transforms.ColorJitter:
    return transforms.ColorJitter(brightness=brightness)


def jitter_contrast(contrast: float = 0.2) -> transforms.ColorJitter:
    return transforms.ColorJitter(contrast=contrast)


def jitter_hue(hue: float = 0.1) -> transforms.ColorJitter:
    return transforms.ColorJitter(hue=hue)


def jitter_saturation(saturation: float = 0.2) -> transforms.ColorJitter:
    return transforms.ColorJitter(saturation=saturation)


def get_phase3_ops(
    brightness: float = 0.2,
    contrast: float = 0.2,
    hue: float = 0.1,
    saturation: float = 0.2,
) -> List[transforms.ColorJitter]:
    return [
        jitter_brightness(brightness=brightness),
        jitter_contrast(contrast=contrast),
        jitter_hue(hue=hue),
        jitter_saturation(saturation=saturation),
    ]


def get_phase3_block(**kwargs) -> transforms.RandomChoice:
    """
    Choose exactly one color perturbation per image.
    """
    return transforms.RandomChoice(get_phase3_ops(**kwargs))


def append_phase3_block(base_ops: list, **kwargs) -> list:
    return list(base_ops) + [get_phase3_block(**kwargs)]


def build_phase3_transform(base_ops: list, **kwargs) -> transforms.Compose:
    return transforms.Compose(append_phase3_block(base_ops, **kwargs))