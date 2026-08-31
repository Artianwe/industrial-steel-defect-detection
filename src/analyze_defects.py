import pandas as pd
import numpy as np

from rle import rle_to_mask


CSV_PATH = "data/processed/train.csv"

df = pd.read_csv(CSV_PATH)

areas = []

print("Analyzing defect sizes...")

for i, row in df.iterrows():

    mask = rle_to_mask(
        row["EncodedPixels"],
        height=256,
        width=1600
    )

    area = int(mask.sum())
    areas.append(area)

areas = np.array(areas)

print("\n========== DEFECT SIZE ANALYSIS ==========")

print("Number of annotations:", len(areas))
print("Minimum area:", areas.min())
print("Maximum area:", areas.max())
print("Mean area:", round(areas.mean(), 2))
print("Median area:", round(np.median(areas), 2))

print("\nPercentiles:")

for percentile in [10, 25, 50, 75, 90, 95, 99]:
    print(
        f"{percentile}th percentile:",
        round(np.percentile(areas, percentile), 2)
    )

print("\nSmall defects:")

for threshold in [500, 1000, 2000, 5000, 10000]:
    count = np.sum(areas < threshold)
    percentage = count / len(areas) * 100

    print(
        f"< {threshold:5d} pixels:",
        f"{count:5d} ({percentage:.2f}%)"
    )