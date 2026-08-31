from dataset import SeverstalDataset
from transforms import preprocess_pair


# ------------------------------------------------------------
# Create dataset
# ------------------------------------------------------------

dataset = SeverstalDataset(
    image_dir="data/processed/train",
    csv_path="data/processed/train.csv",
    transform=preprocess_pair
)


# ------------------------------------------------------------
# Basic dataset information
# ------------------------------------------------------------

print("\nDataset length:", len(dataset))


# ------------------------------------------------------------
# Test first 5 samples
# ------------------------------------------------------------

for index in range(5):

    image, mask, image_id = dataset[index]

    print(f"\nImage: {image_id}")

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Mask shape:",
        mask.shape
    )

    print(
        "Mask classes:",
        mask.unique().tolist()
    )

    print(
        "Image range:",
        round(image.min().item(), 3),
        "to",
        round(image.max().item(), 3)
    )

    print(
        "Defect pixels:",
        int((mask > 0).sum().item())
    )