import pandas as pd
import numpy as np
from PIL import Image
import cv2

from rle import rle_to_mask


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = "data/raw/train.csv"
IMAGE_DIR = "data/raw/train_images"

RESOLUTIONS = [
    (1600, 256),
    (1024, 256),
    (800, 128),
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

unique_images = df["ImageId"].unique()

print(f"Total images: {len(unique_images)}")
print(f"Total annotations: {len(df)}")


# ============================================================
# FUNCTION: CREATE COMPLETE MASK
# ============================================================

def create_complete_mask(image_id):

    image_path = f"{IMAGE_DIR}/{image_id}"

    image = np.array(
        Image.open(image_path).convert("RGB")
    )

    height, width = image.shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    rows = df[df["ImageId"] == image_id]

    for _, row in rows.iterrows():

        class_id = int(row["ClassId"])

        rle_mask = rle_to_mask(
            row["EncodedPixels"],
            height,
            width
        )

        mask[rle_mask == 1] = class_id

    return mask


# ============================================================
# RANDOM SAMPLE
# ============================================================

np.random.seed(42)

sample_size = min(500, len(unique_images))

sample_images = np.random.choice(
    unique_images,
    sample_size,
    replace=False
)


# ============================================================
# STATISTICS STORAGE
# ============================================================

results = {}


for resolution in RESOLUTIONS:

    width, height = resolution

    results[resolution] = {
        "total_original_pixels": 0,
        "total_resized_pixels": 0,
        "images_with_defects": 0,
        "images_with_empty_masks": 0,
    }


# ============================================================
# PROCESS IMAGES
# ============================================================

for counter, image_id in enumerate(sample_images):

    mask = create_complete_mask(image_id)

    original_defect_pixels = np.count_nonzero(mask)

    if original_defect_pixels == 0:
        continue

    for resolution in RESOLUTIONS:

        width, height = resolution

        resized_mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

        resized_defect_pixels = np.count_nonzero(
            resized_mask
        )

        results[resolution]["total_original_pixels"] += (
            original_defect_pixels
        )

        results[resolution]["total_resized_pixels"] += (
            resized_defect_pixels
        )

        if resized_defect_pixels > 0:
            results[resolution]["images_with_defects"] += 1
        else:
            results[resolution]["images_with_empty_masks"] += 1

    if (counter + 1) % 50 == 0:
        print(
            f"Processed {counter + 1}/{sample_size}"
        )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 65)
print("RESOLUTION COMPARISON")
print("=" * 65)

for resolution in RESOLUTIONS:

    width, height = resolution

    original_pixels = results[resolution][
        "total_original_pixels"
    ]

    resized_pixels = results[resolution][
        "total_resized_pixels"
    ]

    empty_masks = results[resolution][
        "images_with_empty_masks"
    ]

    images_with_defects = results[resolution][
        "images_with_defects"
    ]

    if original_pixels > 0:

        # Expected pixel scaling
        original_area = 1600 * 256
        resized_area = width * height

        expected_pixels = (
            original_pixels
            * resized_area
            / original_area
        )

        retention = (
            resized_pixels
            / expected_pixels
        ) * 100

    else:

        retention = 0

    print()
    print(f"Resolution: {width} × {height}")
    print("-" * 40)

    print(
        f"Images retaining defects: "
        f"{images_with_defects}/{sample_size}"
    )

    print(
        f"Images becoming empty: "
        f"{empty_masks}"
    )

    print(
        f"Expected scaled defect pixels: "
        f"{expected_pixels:.0f}"
    )

    print(
        f"Actual resized defect pixels: "
        f"{resized_pixels}"
    )

    print(
        f"Approx. mask retention: "
        f"{retention:.2f}%"
    )


print()
print("=" * 65)
print("COMPARISON COMPLETE")
print("=" * 65)