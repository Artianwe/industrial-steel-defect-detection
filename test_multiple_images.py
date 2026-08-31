from pathlib import Path
import pandas as pd
import subprocess
import sys

CSV_PATH = Path("data/processed/val.csv")
VAL_DIR = Path("data/processed/val")
PREDICT_SCRIPT = Path("src/predict.py")

# Pick representative images containing each defect class.
df = pd.read_csv(CSV_PATH)

selected = {}

for class_id in [1, 2, 3, 4]:
    rows = df[df["ClassId"] == class_id]
    for image_id in rows["ImageId"]:
        image_path = VAL_DIR / image_id
        if image_path.exists():
            selected[class_id] = image_path
            break

# Also choose one image containing multiple defect classes, if available.
for image_id, group in df.groupby("ImageId"):
    classes = set(group["ClassId"].astype(int))
    if len(classes & {1, 2, 3, 4}) >= 2:
        image_path = VAL_DIR / image_id
        if image_path.exists():
            selected["multi"] = image_path
            break

print("=" * 70)
print("SELECTED VALIDATION IMAGES FOR FINAL INFERENCE TEST")
print("=" * 70)

for key, path in selected.items():
    print(f"Test {key}: {path}")

print()

if not selected:
    raise RuntimeError("No validation images were found.")

# Run the existing inference script on every selected image.
for key, path in selected.items():
    print()
    print("=" * 70)
    print(f"RUNNING TEST: {key} -> {path.name}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, str(PREDICT_SCRIPT), "--image", str(path)],
        text=True
    )

    if result.returncode != 0:
        print(f"WARNING: inference failed for {path}")
    else:
        print(f"Completed: {path.name}")

print()
print("=" * 70)
print("FINAL MULTI-IMAGE TEST COMPLETE")
print("=" * 70)
print("Check runs/predictions/ for the generated *_overlay.png files.")
