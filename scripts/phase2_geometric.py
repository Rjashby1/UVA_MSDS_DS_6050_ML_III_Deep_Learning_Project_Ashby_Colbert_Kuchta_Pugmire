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
    return [
        random_horizontal_flip(p=hflip_p),
        random_vertical_flip(p=vflip_p),
        random_rotation(degrees=rotation_degrees),
        random_affine_shear(
            degrees=affine_degrees,
            shear=affine_shear,
            translate=affine_translate,
            scale=affine_scale,
        ),
    ]


def get_phase2_random_choice(**kwargs) -> transforms.RandomChoice:
    return transforms.RandomChoice(get_phase2_ops(**kwargs))


def append_phase2_random_choice(base_ops: list, **kwargs) -> list:
    return list(base_ops) + [get_phase2_random_choice(**kwargs)]


def build_phase2_transform(base_ops: list, **kwargs) -> transforms.Compose:
    return transforms.Compose(append_phase2_random_choice(base_ops, **kwargs))