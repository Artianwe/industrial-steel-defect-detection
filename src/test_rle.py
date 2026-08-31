import pandas as pd
from PIL import Image
from rle import rle_to_mask, mask_area, defect_percentage


# Load annotations
df = pd.read_csv("data/raw/train.csv")

# Select the first annotated defect
row = df.iloc[0]

image_id = row["ImageId"]
class_id = row["ClassId"]
rle = row["EncodedPixels"]

# Load image
image_path = f"data/raw/train_images/{image_id}"
image = Image.open(image_path)

width, height = image.size

# Decode RLE
mask = rle_to_mask(
    rle,
    height=height,
    width=width
)

print("Image:", image_id)
print("Defect class:", class_id)
print("Image dimensions:", image.size)
print("Mask dimensions:", mask.shape)
print("Defective pixels:", mask_area(mask))
print("Defect percentage:", f"{defect_percentage(mask):.4f}%")
print("Unique mask values:", set(mask.flatten()))