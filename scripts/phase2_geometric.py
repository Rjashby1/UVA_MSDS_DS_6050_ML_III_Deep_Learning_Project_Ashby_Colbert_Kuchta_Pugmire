from typing import List

from torchvision import transforms


def horizontal_flip(p: float = 1.0) -> transforms.RandomHorizontalFlip:
    return transforms.RandomHorizontalFlip(p=p)


def vertical_flip(p: float = 1.0) -> transforms.RandomVerticalFlip:
    return transforms.RandomVerticalFlip(p=p)


def random_rotation(degrees: float = 90) -> transforms.RandomRotation:
    return transforms.RandomRotation(degrees=degrees)


def random_affine_shear(
    degrees: float = 0,
    shear=(-10, 10),
    translate=None,
    scale=None,
) -> transforms.RandomAffine:
    return transforms.RandomAffine(
        degrees=degrees,
        shear=shear,
        translate=translate,
        scale=scale,
    )


def get_phase2_ops(
    rotation_degrees: float = 90,
    affine_degrees: float = 0,
    affine_shear=(-10, 10),
    affine_translate=None,
    affine_scale=None,
) -> List:
    return [
        horizontal_flip(),
        vertical_flip(),
        random_rotation(degrees=rotation_degrees),
        random_affine_shear(degrees=affine_degrees,
                     shear=affine_shear,
                     translate=affine_translate,
                     scale=affine_scale,
                    )
        ]


def get_phase2_block(**kwargs):
    """
    Choose exactly one geometric transformation per image.
    """
    return transforms.RandomChoice(get_phase2_ops(**kwargs))

def append_phase2_block(base_ops: list, **kwargs) -> list:
    ops = list(base_ops)
    if len(ops) < 1:
        raise ValueError("base_ops must contain at least one transform.")
    return [ops[0], get_phase2_block(**kwargs), *ops[1:]]


def build_phase2_transform(base_ops: list, **kwargs) -> transforms.Compose:
    return transforms.Compose(append_phase2_block(base_ops, **kwargs))