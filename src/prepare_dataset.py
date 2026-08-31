import os
import shutil
import pandas as pd
from PIL import Image


# ============================================================
# Configuration
# ============================================================

CSV_PATH = "data/raw/train.csv"
IMAGE_DIR = "data/raw/train_images"

OUTPUT_DIR = "data/processed"

TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")


# ============================================================
# Create output directories
# ============================================================

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)


# ============================================================
# Load annotations
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Total annotation rows:", len(df))
print("Unique images:", df["ImageId"].nunique())


# ============================================================
# Get unique images
# ============================================================

image_ids = df["ImageId"].unique()

print("Images to process:", len(image_ids))


# ============================================================
# Train / validation split
# ============================================================

# Fixed random seed makes the split reproducible
train_images = pd.Series(image_ids).sample(
    frac=0.8,
    random_state=42
)

train_set = set(train_images)

val_images = [
    image_id
    for image_id in image_ids
    if image_id not in train_set
]


print("Training images:", len(train_images))
print("Validation images:", len(val_images))


# ============================================================
# Copy images
# ============================================================

def copy_images(image_list, destination):

    copied = 0

    for image_id in image_list:

        source = os.path.join(
            IMAGE_DIR,
            image_id
        )

        target = os.path.join(
            destination,
            image_id
        )

        if os.path.exists(source):
            shutil.copy2(source, target)
            copied += 1

    return copied


train_count = copy_images(
    train_images,
    TRAIN_DIR
)

val_count = copy_images(
    val_images,
    VAL_DIR
)


# ============================================================
# Save annotation splits
# ============================================================

train_df = df[
    df["ImageId"].isin(train_set)
]

val_df = df[
    df["ImageId"].isin(val_images)
]


train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False
)

val_df.to_csv(
    os.path.join(OUTPUT_DIR, "val.csv"),
    index=False
)


# ============================================================
# Verification
# ============================================================

print()
print("========== PREPROCESSING COMPLETE ==========")
print("Training images copied:", train_count)
print("Validation images copied:", val_count)
print("Training annotations:", len(train_df))
print("Validation annotations:", len(val_df))

print()
print("Output structure:")
print("data/processed/")
print("├── train/")
print("├── val/")
print("├── train.csv")
print("└── val.csv")