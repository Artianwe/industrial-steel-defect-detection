import matplotlib.pyplot as plt
import numpy as np

from dataset import SeverstalDataset
from transforms import resize_pair


# ------------------------------------------------------------
# Create dataset WITHOUT normalization
# ------------------------------------------------------------

dataset = SeverstalDataset(
    image_dir="data/processed/train",
    csv_path="data/processed/train.csv",
    transform=None
)


# ------------------------------------------------------------
# Select specific samples
# ------------------------------------------------------------

sample_indices = [0, 1, 2, 3, 4]


# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    len(sample_indices),
    3,
    figsize=(18, 14)
)


for row_number, index in enumerate(sample_indices):

    # --------------------------------------------------------
    # Load complete sample from dataset
    # --------------------------------------------------------

    image_tensor, mask_tensor, image_id = dataset[index]

    # --------------------------------------------------------
    # Convert image tensor to display format
    # --------------------------------------------------------

    image = (
        image_tensor
        .permute(1, 2, 0)
        .numpy()
    )

    # --------------------------------------------------------
    # Get complete ground-truth mask
    # --------------------------------------------------------

    original_mask = (
        mask_tensor
        .squeeze(0)
        .numpy()
    )

    # --------------------------------------------------------
    # Resize image + COMPLETE mask
    # --------------------------------------------------------

    resized_image, resized_mask = resize_pair(
        image_tensor,
        mask_tensor
    )

    resized_image = (
        resized_image
        .permute(1, 2, 0)
        .numpy()
    )

    resized_mask = (
        resized_mask
        .squeeze(0)
        .numpy()
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    axes[row_number, 0].imshow(image)

    axes[row_number, 0].set_title(
        f"{image_id}\n"
        f"Original: 1600 × 256"
    )

    axes[row_number, 0].axis("off")

    # --------------------------------------------------------
    # Resized complete mask
    # --------------------------------------------------------

    axes[row_number, 1].imshow(
        resized_mask,
        cmap="nipy_spectral",
        vmin=0,
        vmax=4
    )

    axes[row_number, 1].set_title(
        f"Complete Ground Truth Mask\n"
        f"800 × 128\n"
        f"Classes: {np.unique(original_mask).tolist()}"
    )

    axes[row_number, 1].axis("off")

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    axes[row_number, 2].imshow(
        resized_image
    )

    axes[row_number, 2].imshow(
        resized_mask,
        cmap="nipy_spectral",
        alpha=0.45,
        vmin=0,
        vmax=4
    )

    axes[row_number, 2].set_title(
        "Complete Ground Truth Overlay"
    )

    axes[row_number, 2].axis("off")


plt.tight_layout()
plt.show()