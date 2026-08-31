import os
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from rle import rle_to_mask


class SeverstalDataset(Dataset):
    """
    PyTorch dataset for Severstal steel defect segmentation.

    Mask values:
        0 = background
        1 = defect class 1
        2 = defect class 2
        3 = defect class 3
        4 = defect class 4

    Returns:
        image    -> Tensor [3, H, W]
        mask     -> Tensor [1, H, W]
        image_id -> image filename
    """

    def __init__(
        self,
        image_dir,
        csv_path,
        transform=None
    ):
        self.image_dir = image_dir
        self.transform = transform

        # ----------------------------------------------------
        # Load annotations
        # ----------------------------------------------------

        annotations = pd.read_csv(csv_path)

        self.annotations = defaultdict(list)

        for _, row in annotations.iterrows():

            image_id = row["ImageId"]
            class_id = int(row["ClassId"])
            rle = row["EncodedPixels"]

            self.annotations[image_id].append(
                (class_id, rle)
            )

        # ----------------------------------------------------
        # Find images that actually exist
        # ----------------------------------------------------

        available_images = set(
            os.listdir(image_dir)
        )

        self.image_ids = [
            image_id
            for image_id in self.annotations.keys()
            if image_id in available_images
        ]

        self.image_ids.sort()

        # ----------------------------------------------------
        # Build sampling information
        # ----------------------------------------------------

        self.image_classes = {}
        self.image_defect_area = {}

        for image_id in self.image_ids:

            classes = set()
            total_area = 0

            for class_id, rle in self.annotations[image_id]:

                classes.add(class_id)

                # Calculate RLE area without constructing
                # a full-resolution mask.
                parts = rle.split()

                for i in range(0, len(parts), 2):

                    start = int(parts[i])
                    length = int(parts[i + 1])

                    total_area += length

            self.image_classes[image_id] = classes
            self.image_defect_area[image_id] = total_area

        print(
            f"Dataset loaded: {len(self.image_ids)} images"
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):

        # ----------------------------------------------------
        # Select image
        # ----------------------------------------------------

        image_id = self.image_ids[index]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image_path = os.path.join(
            self.image_dir,
            image_id
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image_array = np.array(image)

        height, width = image_array.shape[:2]

        # ----------------------------------------------------
        # Create empty multi-class mask
        # ----------------------------------------------------

        combined_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        # ----------------------------------------------------
        # Decode annotations
        # ----------------------------------------------------

        for class_id, rle in self.annotations[image_id]:

            mask = rle_to_mask(
                rle,
                height=height,
                width=width
            )

            combined_mask[mask == 1] = class_id

        # ----------------------------------------------------
        # Convert image
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            image_array
        ).permute(
            2,
            0,
            1
        ).float() / 255.0

        # ----------------------------------------------------
        # Convert mask
        # ----------------------------------------------------

        mask_tensor = torch.from_numpy(
            combined_mask
        ).unsqueeze(0).long()

        # ----------------------------------------------------
        # Apply preprocessing
        # ----------------------------------------------------

        if self.transform is not None:

            image_tensor, mask_tensor = self.transform(
                image_tensor,
                mask_tensor
            )

        return (
            image_tensor,
            mask_tensor,
            image_id
        )

    # ========================================================
    # DEFECT-AWARE SAMPLING
    # ========================================================

    def get_sampling_weights(self):
        """
        Generate image sampling weights.

        The purpose is to expose the model more frequently to:

        1. Class 1 defects
        2. Class 2 defects
        3. Small defects

        This does NOT duplicate files on disk.

        It only changes the probability that an existing
        training image is selected during an epoch.
        """

        weights = []

        for image_id in self.image_ids:

            classes = self.image_classes[image_id]
            defect_area = self.image_defect_area[image_id]

            # ------------------------------------------------
            # Start with normal probability.
            # ------------------------------------------------

            weight = 1.0

            # ------------------------------------------------
            # Class 1 and Class 2
            # ------------------------------------------------

            if 1 in classes:
                weight *= 1.8

            if 2 in classes:
                weight *= 1.8

            # ------------------------------------------------
            # Small-defect emphasis
            #
            # These thresholds are based on the defect-area
            # analysis already performed on this dataset.
            # ------------------------------------------------

            if defect_area < 10000:
                weight *= 1.5

            if defect_area < 5000:
                weight *= 1.25

            # ------------------------------------------------
            # Prevent extreme sampling weights.
            # ------------------------------------------------

            weight = min(
                weight,
                5.0
            )

            weights.append(weight)

        return torch.tensor(
            weights,
            dtype=torch.double
        )

    def get_sampling_statistics(self):
        """
        Return statistics describing the sampling strategy.
        """

        weights = self.get_sampling_weights()

        return {
            "min_weight": weights.min().item(),
            "max_weight": weights.max().item(),
            "mean_weight": weights.mean().item(),
            "num_images": len(weights)
        }