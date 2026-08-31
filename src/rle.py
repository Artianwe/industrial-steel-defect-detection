import numpy as np


def rle_to_mask(rle, height=256, width=1600):
    """
    Convert a Severstal RLE-encoded defect mask into a binary mask.

    Parameters
    ----------
    rle : str
        Run-Length Encoded pixel information.
    height : int
        Image height.
    width : int
        Image width.

    Returns
    -------
    np.ndarray
        Binary defect mask with shape (height, width).
    """

    mask = np.zeros(height * width, dtype=np.uint8)

    if not isinstance(rle, str) or not rle.strip():
        return mask.reshape((height, width))

    values = list(map(int, rle.split()))

    starts = np.asarray(values[0::2]) - 1
    lengths = np.asarray(values[1::2])

    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1

    # Severstal stores pixels in column-major order
    mask = mask.reshape((width, height)).T

    return mask


def mask_area(mask):
    """
    Calculate the number of defective pixels.
    """
    return int(mask.sum())


def defect_percentage(mask):
    """
    Calculate the percentage of the image occupied by the defect.
    """
    return float(mask.mean() * 100)