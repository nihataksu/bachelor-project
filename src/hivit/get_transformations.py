from torchvision import transforms


def read_bool(val):
    return val is True or val.lower() == "true" or val.lower() == "yes"


def get_transforms(parameters):
    transformations = []

    if read_bool(parameters.TRANSFORM_RANDOM_HORIZONTAL_FLIP_ENABLED):
        transforms.append(transforms.RandomHorizontalFlip())
    if read_bool(parameters.TRANSFORM_RANDOM_ROTATION_ENABLED):
        transforms.append(
            transforms.RandomRotation(parameters.TRANSFORM_RANDOM_ROTATION_DEGREE)
        )
    if read_bool(parameters.TRANSFORM_RANDOM_CROP_ENABLED):
        transforms.append(
            transforms.RandomCrop(parameters.IMAGE_SIZE),
            padding=parameters.TRANSFORM_RANDOM_CROP_PADDING,
        )
    if read_bool(parameters.TRANSFORM_COLOR_JITTER_ENABLED):
        transforms.append(
            transforms.ColorJitter(
                brightness=parameters.TRANSFORM_COLOR_JITTER_BRIGHTNESS,
                contrast=parameters.TRANSFORM_COLOR_JITTER_CONTRAST,
                saturation=parameters.TRANSFORM_COLOR_JITTER_SATURATION,
                hue=parameters.TRANSFORM_COLOR_JITTER_HUE,
            )
        )

    if read_bool(parameters.TRANSFORM_RANDOM_ERASING):
        transforms.append(
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"
            )
        )

    return transformations
