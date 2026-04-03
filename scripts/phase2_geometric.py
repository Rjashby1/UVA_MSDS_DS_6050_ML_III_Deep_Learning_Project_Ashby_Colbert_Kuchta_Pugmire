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
        "rotation" : transforms.RandomApply(
            [random_rotation(degrees=rotation_degrees)],
            p=0.5),
        "affine" : transforms.RandomApply([random_affine_shear(
                   degrees=affine_degrees,
                   shear=affine_shear,
                   translate=affine_translate,
                   scale=affine_scale,
                    )], p=0.3),
        }


def get_phase2_block(mode: str = "compose", transform_name: str | None = None, **kwargs):
    """
    Build the geometric augmentation block.

    Parameters
    ----------
    mode : str
        "random" to apply one randomly chosen transform,
        "compose" to compose all transforms.
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
    elif mode == "compose":
        return transforms.Compose(op_list)
    else:
        raise ValueError("Invalid mode. Select random or compose or choose a specific transformation")


def append_phase2_block(base_ops: list, **kwargs) -> list:
    ops = list(base_ops)
    if len(ops) < 1:
        raise ValueError("Expected base_ops to contain at least Resize, ToTensor, and Normalize.")
    return [ops[0], get_phase2_block(**kwargs), *ops[1:]]


def build_phase2_transform(base_ops: list, **kwargs) -> transforms.Compose:
    return transforms.Compose(append_phase2_block(base_ops, **kwargs))