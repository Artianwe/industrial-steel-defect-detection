from dataset import SeverstalDataset
from transforms import preprocess_pair


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

dataset = SeverstalDataset(
    image_dir="data/processed/val",
    csv_path="data/processed/val.csv",
    transform=preprocess_pair
)


# ============================================================
# CHECK FIRST IMAGE
# ============================================================

image, mask, image_id = dataset[0]

print("=" * 60)
print("DATASET CHECK")
print("=" * 60)

print()
print("Image:", image_id)

print(
    "Image shape:",
    image.shape
)

print(
    "Mask shape:",
    mask.shape
)

print(
    "Image range:",
    image.min().item(),
    "to",
    image.max().item()
)

print(
    "Mask classes:",
    mask.unique().tolist()
)

print()
print("Class pixel counts:")

for c in mask.unique():

    count = (
        mask == c
    ).sum().item()

    print(
        f"Class {int(c)}: {count:,}"
    )

print()
print("=" * 60)