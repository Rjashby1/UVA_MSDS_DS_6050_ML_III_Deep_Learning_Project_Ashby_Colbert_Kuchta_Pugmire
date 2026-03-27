from torchvision import transforms


def random_horizontal_flip(p: float = 0.5) -> transforms.RandomHorizontalFlip:
    return transforms.RandomHorizontalFlip(p=p)


def random_vertical_flip(p: float = 0.5) -> transforms.RandomVerticalFlip:
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
    hflip_p: float = 0.5,
    vflip_p: float = 0.5,
    rotation_degrees: float = 90,
    affine_degrees: float = 0,
    affine_shear=(-10, 10),
    affine_translate=None,
    affine_scale=None,
):
    return {
        "hflip" : random_horizontal_flip(p=hflip_p),
        "vflip" : random_vertical_flip(p=vflip_p),
        "rotation" : random_rotation(degrees=rotation_degrees),
        "affine" : random_affine_shear(
                   degrees=affine_degrees,
                   shear=affine_shear,
                   translate=affine_translate,
                   scale=affine_scale,
                    ),
        }


def get_phase2_block(mode: str = "all", transform_name: str | None = None, **kwargs):
    """
    Build the geometric augmentation block.

    Parameters
    ----------
    mode : str
        "random" to apply one randomly chosen transform,
        "all" to apply all geometric transforms sequentially.
    transform_name : str | None
        If provided, applies only the named transform and ignores mode.

    Returns
    -------
    torchvision transform
        A composed transform block.
    """

    ops = get_phase2_ops(**kwargs)

    if transform_name is not None:
        if transform_name not in ops:
            raise ValueError(f"Unknown transformation {transform_name}. Choose from {list(ops.keys())}")
        return transforms.Compose([ops[transform_name]])
    
    op_list = list(ops.values())

    if mode == "random":
        return transforms.RandomChoice(op_list)
    elif mode == "all":
        return transforms.Compose(op_list)
    else:
        raise ValueError("Invalid mode. Select random or all or choose a specific transformation")


def append_phase2_block(base_ops: list, **kwargs) -> list:
    return list(base_ops) + [get_phase2_block(**kwargs)]


def build_phase2_transform(base_ops: list, **kwargs) -> transforms.Compose:
    return transforms.Compose(append_phase2_block(base_ops, **kwargs))