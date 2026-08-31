import torch
from torch.utils.data import DataLoader

from dataset import SeverstalDataset
from transforms import preprocess_pair


# ============================================================
# SETTINGS
# ============================================================

IMAGE_DIR = "data/processed/train"
CSV_PATH = "data/processed/train.csv"

BATCH_SIZE = 2
NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DATALOADER TEST")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# DATASET
# ============================================================

dataset = SeverstalDataset(
    image_dir=IMAGE_DIR,
    csv_path=CSV_PATH,
    transform=preprocess_pair
)


print(f"Dataset size: {len(dataset)}")


# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


print(f"Batch size: {BATCH_SIZE}")
print(f"Number of batches: {len(loader)}")


# ============================================================
# GET ONE BATCH
# ============================================================

images, masks, image_ids = next(iter(loader))


print("\nBatch information")
print("-" * 40)

print(
    "Images:",
    images.shape
)

print(
    "Masks:",
    masks.shape
)

print(
    "Image IDs:",
    image_ids
)

print(
    "Image dtype:",
    images.dtype
)

print(
    "Mask dtype:",
    masks.dtype
)


# ============================================================
# CHECK VALUES
# ============================================================

print("\nMask class check")
print("-" * 40)

unique_classes = torch.unique(masks)

print(
    "Classes in batch:",
    unique_classes.tolist()
)

print(
    "Image min:",
    images.min().item()
)

print(
    "Image max:",
    images.max().item()
)


# ============================================================
# VALIDATION CHECKS
# ============================================================

assert images.ndim == 4
assert masks.ndim == 4

assert images.shape[0] == BATCH_SIZE
assert masks.shape[0] == BATCH_SIZE

assert images.shape[1] == 3
assert masks.shape[1] == 1

assert images.shape[2] == 256
assert images.shape[3] == 1024

assert masks.shape[2] == 256
assert masks.shape[3] == 1024

assert masks.dtype == torch.long

assert torch.all(
    (masks >= 0) & (masks <= 4)
)

print("\nAll DataLoader checks PASSED")

print("=" * 60)