import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SeverstalDataset
from transforms import preprocess_pair
from model import create_model


# ============================================================
# CONFIG
# ============================================================

NUM_CLASSES = 5
BATCH_SIZE = 2

VAL_IMAGE_DIR = "data/processed/val"
VAL_CSV = "data/processed/val.csv"

MODEL_PATH = "runs/segmentation_sampling/best_model.pth"

OUTPUT_DIR = "runs/segmentation/baseline_verification"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("BASELINE EVALUATION VERIFICATION")
print("=" * 70)

print()
print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# DATASET
# ============================================================

print()
print("-" * 70)
print("Loading validation dataset")
print("-" * 70)

dataset = SeverstalDataset(
    image_dir=VAL_IMAGE_DIR,
    csv_path=VAL_CSV,
    transform=preprocess_pair
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(
    f"Validation dataset size: {len(dataset)}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Number of batches: {len(loader)}"
)


# ============================================================
# MODEL
# ============================================================

print()
print("-" * 70)
print("Loading baseline model")
print("-" * 70)

model = create_model(
    num_classes=NUM_CLASSES,
    pretrained=False
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

state_dict = checkpoint["model_state_dict"]

model.load_state_dict(
    state_dict,
    strict=True
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")

print(
    f"Checkpoint epoch: {checkpoint['epoch']}"
)

print(
    f"Checkpoint best_miou: "
    f"{checkpoint['best_miou']:.8f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion_matrix = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)


# ============================================================
# METRIC STORAGE
# ============================================================

intersection = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

union = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

pred_count = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

gt_count = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

tp = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

fp = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

fn = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)


# ============================================================
# EVALUATION
# ============================================================

print()
print("-" * 70)
print("Running baseline evaluation")
print("-" * 70)

with torch.no_grad():

    for batch_idx, batch in enumerate(loader):

        images = batch[0]
        masks = batch[1]

        images = images.to(device)

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        masks = masks.to(device).long()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # ----------------------------------------------------
        # Pixel statistics
        # ----------------------------------------------------

        gt = masks.cpu().numpy()
        pred = predictions.cpu().numpy()

        for b in range(gt.shape[0]):

            gt_img = gt[b]
            pred_img = pred[b]

            # ------------------------------------------------
            # Confusion matrix
            # ------------------------------------------------

            flat_gt = gt_img.reshape(-1)
            flat_pred = pred_img.reshape(-1)

            for gt_class in range(NUM_CLASSES):

                gt_mask = (
                    flat_gt == gt_class
                )

                if not np.any(gt_mask):
                    continue

                predicted_for_gt = (
                    flat_pred[gt_mask]
                )

                counts = np.bincount(
                    predicted_for_gt,
                    minlength=NUM_CLASSES
                )

                confusion_matrix[
                    gt_class
                ] += counts

            # ------------------------------------------------
            # Per-class statistics
            # ------------------------------------------------

            for class_id in range(NUM_CLASSES):

                gt_mask = (
                    gt_img == class_id
                )

                pred_mask = (
                    pred_img == class_id
                )

                gt_pixels = int(
                    gt_mask.sum()
                )

                pred_pixels = int(
                    pred_mask.sum()
                )

                inter = int(
                    np.logical_and(
                        gt_mask,
                        pred_mask
                    ).sum()
                )

                union_pixels = (
                    gt_pixels
                    +
                    pred_pixels
                    -
                    inter
                )

                gt_count[class_id] += (
                    gt_pixels
                )

                pred_count[class_id] += (
                    pred_pixels
                )

                intersection[class_id] += (
                    inter
                )

                union[class_id] += (
                    union_pixels
                )

                tp[class_id] += inter

                fp[class_id] += (
                    pred_pixels - inter
                )

                fn[class_id] += (
                    gt_pixels - inter
                )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        processed = min(
            (batch_idx + 1) * BATCH_SIZE,
            len(dataset)
        )

        if (
            processed % 200 == 0
            or processed == len(dataset)
        ):

            print(
                f"Processed "
                f"{processed}/{len(dataset)} images"
            )


# ============================================================
# METRICS
# ============================================================

iou = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

dice = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

precision = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

recall = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

for class_id in range(NUM_CLASSES):

    if union[class_id] > 0:

        iou[class_id] = (
            intersection[class_id]
            /
            union[class_id]
        )

    dice_denominator = (
        2 * tp[class_id]
        +
        fp[class_id]
        +
        fn[class_id]
    )

    if dice_denominator > 0:

        dice[class_id] = (
            2 * tp[class_id]
            /
            dice_denominator
        )

    precision_denominator = (
        tp[class_id] + fp[class_id]
    )

    if precision_denominator > 0:

        precision[class_id] = (
            tp[class_id]
            /
            precision_denominator
        )

    recall_denominator = (
        tp[class_id] + fn[class_id]
    )

    if recall_denominator > 0:

        recall[class_id] = (
            tp[class_id]
            /
            recall_denominator
        )


# ============================================================
# MEAN METRICS
# ============================================================

mean_iou = np.mean(iou)

mean_dice = np.mean(dice)

defect_mean_iou = np.mean(
    iou[1:]
)

defect_mean_dice = np.mean(
    dice[1:]
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("VERIFICATION RESULTS")
print("=" * 70)

print()

print(
    f"Mean IoU:        {mean_iou:.5f}"
)

print(
    f"Mean Dice:       {mean_dice:.5f}"
)

print(
    f"Defect Mean IoU: {defect_mean_iou:.5f}"
)

print(
    f"Defect Mean Dice:{defect_mean_dice:.5f}"
)


# ============================================================
# PER CLASS
# ============================================================

print()
print("-" * 70)
print("PER-CLASS RESULTS")
print("-" * 70)

for class_id in range(NUM_CLASSES):

    print(
        f"Class {class_id} | "
        f"IoU: {iou[class_id]:.5f} | "
        f"Dice: {dice[class_id]:.5f} | "
        f"Precision: {precision[class_id]:.5f} | "
        f"Recall: {recall[class_id]:.5f}"
    )


# ============================================================
# PIXEL COUNTS
# ============================================================

print()
print("-" * 70)
print("PIXEL COUNTS")
print("-" * 70)

for class_id in range(NUM_CLASSES):

    print(
        f"Class {class_id} | "
        f"GT: {gt_count[class_id]:,} | "
        f"Pred: {pred_count[class_id]:,}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()
print("Rows = Ground Truth")
print("Columns = Prediction")
print()

print(confusion_matrix)


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

normalized = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.float64
)

for row in range(NUM_CLASSES):

    total = confusion_matrix[row].sum()

    if total > 0:

        normalized[row] = (
            confusion_matrix[row]
            /
            total
            *
            100
        )

print()
print("=" * 70)
print("NORMALIZED CONFUSION MATRIX (%)")
print("=" * 70)

print()

for row in range(NUM_CLASSES):

    values = " | ".join(
        f"{normalized[row, col]:7.3f}%"
        for col in range(NUM_CLASSES)
    )

    print(
        f"GT Class {row}: {values}"
    )


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = os.path.join(
    OUTPUT_DIR,
    "metrics.csv"
)

with open(
    metrics_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "class",
        "gt_pixels",
        "pred_pixels",
        "iou",
        "dice",
        "precision",
        "recall"
    ])

    for class_id in range(NUM_CLASSES):

        writer.writerow([
            class_id,
            gt_count[class_id],
            pred_count[class_id],
            iou[class_id],
            dice[class_id],
            precision[class_id],
            recall[class_id]
        ])


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.csv"
)

np.savetxt(
    confusion_path,
    confusion_matrix,
    delimiter=",",
    fmt="%d"
)


# ============================================================
# SAVE NORMALIZED MATRIX
# ============================================================

normalized_path = os.path.join(
    OUTPUT_DIR,
    "normalized_confusion_matrix.csv"
)

np.savetxt(
    normalized_path,
    normalized,
    delimiter=",",
    fmt="%.6f"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("BASELINE VERIFICATION COMPLETE")
print("=" * 70)

print()
print(f"Metrics: {metrics_path}")
print(f"Confusion matrix: {confusion_path}")
print(f"Normalized matrix: {normalized_path}")