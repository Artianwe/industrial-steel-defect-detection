import torch
import torch.nn.functional as F


TARGET_HEIGHT = 256
TARGET_WIDTH = 1024


def resize_pair(image, mask):
    """
    Resize an image and its segmentation mask.

    Image:
        Bilinear interpolation

    Mask:
        Nearest-neighbor interpolation

    This preserves the discrete class labels in the mask.
    """

    image = image.unsqueeze(0)
    mask = mask.unsqueeze(0)

    image = F.interpolate(
        image,
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        mode="bilinear",
        align_corners=False
    )

    mask = F.interpolate(
        mask.float(),
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        mode="nearest"
    )

    image = image.squeeze(0)
    mask = mask.squeeze(0).long()

    return image, mask


def normalize_image(image):
    """
    Normalize image values from [0, 1]
    using ImageNet mean and standard deviation.
    """

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    return (image - mean) / std


def preprocess_pair(image, mask):
    """
    Complete deterministic preprocessing pipeline.
    """

    image, mask = resize_pair(
        image,
        mask
    )

    image = normalize_image(image)

    return image, mask