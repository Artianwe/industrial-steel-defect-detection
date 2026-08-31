import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SeverstalDataset
from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5
BATCH_SIZE = 2

VAL_IMAGE_DIR = "data/processed/val"
VAL_CSV = "data/processed/val.csv"

MODEL_PATH = "runs/segmentation/best_model.pth"

OUTPUT_DIR = "runs/segmentation/diagnosis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("BASELINE MODEL DIAGNOSTIC ANALYSIS")
print("=" * 70)

print()
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# DATASET
# ============================================================

print()
print("-" * 70)
print("Loading validation dataset")
print("-" * 70)

dataset = SeverstalDataset(
    image_dir=VAL_IMAGE_DIR,
    csv_path=VAL_CSV
)

print(f"Validation dataset: {len(dataset)} images")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(f"Batch size: {BATCH_SIZE}")
print(f"Number of batches: {len(loader)}")


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

# Handle either a raw state_dict or a checkpoint dictionary
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

print("Baseline model loaded successfully.")


# ============================================================
# STATISTICS STORAGE
# ============================================================

# Confusion matrix:
# rows = ground truth
# columns = prediction

confusion_matrix = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)

# Ground truth and prediction pixel counts
gt_pixel_count = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

pred_pixel_count = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

# TP / FP / FN
true_positive = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

false_positive = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

false_negative = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

# Image-level statistics
images_with_class = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

images_predicted_class = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

images_completely_missed = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

# Defect pixels
total_gt_defect_pixels = 0
total_pred_defect_pixels = 0
total_missed_defect_pixels = 0


# ============================================================
# CLASS NAMES
# ============================================================

class_names = {
    0: "Background",
    1: "Defect Class 1",
    2: "Defect Class 2",
    3: "Defect Class 3",
    4: "Defect Class 4"
}


# ============================================================
# EVALUATION
# ============================================================

print()
print("-" * 70)
print("Running baseline diagnostic")
print("-" * 70)

with torch.no_grad():

    for batch_idx, batch in enumerate(loader):

        # ----------------------------------------------------
        # Support common dataset return formats
        # ----------------------------------------------------

        if isinstance(batch, (tuple, list)):

            images = batch[0]
            masks = batch[1]

        elif isinstance(batch, dict):

            images = batch["image"]
            masks = batch["mask"]

        else:
            raise TypeError(
                "Unsupported dataset output format."
            )

        images = images.to(device)

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        masks = masks.to(device).long()

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # ----------------------------------------------------
        # Move to CPU
        # ----------------------------------------------------

        gt = masks.cpu().numpy()
        pred = predictions.cpu().numpy()

        # ----------------------------------------------------
        # Process each image individually
        # ----------------------------------------------------

        for i in range(gt.shape[0]):

            gt_image = gt[i]
            pred_image = pred[i]

            # ------------------------------------------------
            # Pixel statistics
            # ------------------------------------------------

            for class_id in range(NUM_CLASSES):

                gt_mask = (
                    gt_image == class_id
                )

                pred_mask = (
                    pred_image == class_id
                )

                gt_count = int(
                    gt_mask.sum()
                )

                pred_count = int(
                    pred_mask.sum()
                )

                gt_pixel_count[class_id] += gt_count
                pred_pixel_count[class_id] += pred_count

                # ------------------------------------------------
                # Image-level presence
                # ------------------------------------------------

                if gt_count > 0:
                    images_with_class[class_id] += 1

                if pred_count > 0:
                    images_predicted_class[class_id] += 1

                # ------------------------------------------------
                # TP / FP / FN
                # ------------------------------------------------

                intersection = int(
                    np.logical_and(
                        gt_mask,
                        pred_mask
                    ).sum()
                )

                true_positive[class_id] += intersection

                false_positive[class_id] += (
                    pred_count - intersection
                )

                false_negative[class_id] += (
                    gt_count - intersection
                )

                # ------------------------------------------------
                # Completely missed class
                # ------------------------------------------------

                if (
                    gt_count > 0
                    and pred_count == 0
                ):
                    images_completely_missed[
                        class_id
                    ] += 1

            # ------------------------------------------------
            # Defect pixels
            # ------------------------------------------------

            gt_defect = (
                gt_image > 0
            )

            pred_defect = (
                pred_image > 0
            )

            gt_defect_count = int(
                gt_defect.sum()
            )

            pred_defect_count = int(
                pred_defect.sum()
            )

            retained = int(
                np.logical_and(
                    gt_defect,
                    pred_defect
                ).sum()
            )

            total_gt_defect_pixels += (
                gt_defect_count
            )

            total_pred_defect_pixels += (
                pred_defect_count
            )

            total_missed_defect_pixels += (
                gt_defect_count - retained
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (batch_idx + 1) % 100 == 0
            or
            (batch_idx + 1) == len(loader)
        ):

            processed = min(
                (batch_idx + 1) * BATCH_SIZE,
                len(dataset)
            )

            print(
                f"Processed {processed}/{len(dataset)}"
            )


# ============================================================
# CALCULATE METRICS
# ============================================================

iou = np.zeros(NUM_CLASSES)
dice = np.zeros(NUM_CLASSES)
precision = np.zeros(NUM_CLASSES)
recall = np.zeros(NUM_CLASSES)

for class_id in range(NUM_CLASSES):

    tp = true_positive[class_id]
    fp = false_positive[class_id]
    fn = false_negative[class_id]

    denominator_iou = (
        tp + fp + fn
    )

    denominator_precision = (
        tp + fp
    )

    denominator_recall = (
        tp + fn
    )

    denominator_dice = (
        2 * tp + fp + fn
    )

    if denominator_iou > 0:
        iou[class_id] = (
            tp / denominator_iou
        )

    if denominator_dice > 0:
        dice[class_id] = (
            2 * tp / denominator_dice
        )

    if denominator_precision > 0:
        precision[class_id] = (
            tp / denominator_precision
        )

    if denominator_recall > 0:
        recall[class_id] = (
            tp / denominator_recall
        )


defect_classes = range(1, NUM_CLASSES)

defect_mean_iou = np.mean(
    iou[1:]
)

defect_mean_dice = np.mean(
    dice[1:]
)

defect_mean_precision = np.mean(
    precision[1:]
)

defect_mean_recall = np.mean(
    recall[1:]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("PIXEL DISTRIBUTION")
print("=" * 70)

print()

for class_id in range(NUM_CLASSES):

    percentage = (
        gt_pixel_count[class_id]
        /
        gt_pixel_count.sum()
        *
        100
    )

    print(
        f"Class {class_id} ({class_names[class_id]}):"
    )

    print(
        f"  Ground-truth pixels: "
        f"{gt_pixel_count[class_id]:,}"
    )

    print(
        f"  Predicted pixels:    "
        f"{pred_pixel_count[class_id]:,}"
    )

    print(
        f"  GT percentage:       "
        f"{percentage:.4f}%"
    )

    print()


# ============================================================
# PER CLASS METRICS
# ============================================================

print("=" * 70)
print("PER-CLASS DIAGNOSTIC METRICS")
print("=" * 70)

print()

for class_id in range(NUM_CLASSES):

    print(
        f"Class {class_id} | "
        f"IoU: {iou[class_id]:.5f} | "
        f"Dice: {dice[class_id]:.5f} | "
        f"Precision: {precision[class_id]:.5f} | "
        f"Recall: {recall[class_id]:.5f}"
    )


# ============================================================
# IMAGE-LEVEL STATISTICS
# ============================================================

print()
print("=" * 70)
print("IMAGE-LEVEL CLASS STATISTICS")
print("=" * 70)

print()

for class_id in range(NUM_CLASSES):

    present = images_with_class[class_id]
    predicted = images_predicted_class[class_id]
    missed = images_completely_missed[class_id]

    if present > 0:

        miss_rate = (
            missed / present * 100
        )

    else:

        miss_rate = 0.0

    print(
        f"Class {class_id} ({class_names[class_id]})"
    )

    print(
        f"  Images containing class: "
        f"{present}"
    )

    print(
        f"  Images predicted class:  "
        f"{predicted}"
    )

    print(
        f"  Completely missed:       "
        f"{missed}"
    )

    print(
        f"  Complete miss rate:       "
        f"{miss_rate:.2f}%"
    )

    print()


# ============================================================
# DEFECT RETENTION
# ============================================================

print("=" * 70)
print("DEFECT PIXEL RETENTION")
print("=" * 70)

retention = (
    (
        total_gt_defect_pixels
        -
        total_missed_defect_pixels
    )
    /
    total_gt_defect_pixels
    *
    100
)

print()
print(
    f"Ground-truth defect pixels: "
    f"{total_gt_defect_pixels:,}"
)

print(
    f"Predicted defect pixels:    "
    f"{total_pred_defect_pixels:,}"
)

print(
    f"Missed defect pixels:       "
    f"{total_missed_defect_pixels:,}"
)

print(
    f"Defect pixel retention:     "
    f"{retention:.2f}%"
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

print(
    confusion_matrix
)


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

normalized_confusion = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.float64
)

for class_id in range(NUM_CLASSES):

    row_total = (
        confusion_matrix[class_id].sum()
    )

    if row_total > 0:

        normalized_confusion[class_id] = (
            confusion_matrix[class_id]
            /
            row_total
            *
            100
        )


print()
print("=" * 70)
print("NORMALIZED CONFUSION MATRIX (%)")
print("=" * 70)

print()
print(
    "Each row shows where the ground-truth "
    "class pixels were predicted."
)

print()

for class_id in range(NUM_CLASSES):

    values = " | ".join(
        f"{normalized_confusion[class_id, j]:6.2f}%"
        for j in range(NUM_CLASSES)
    )

    print(
        f"GT Class {class_id}: {values}"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

print()

print(
    f"Defect Mean IoU:       "
    f"{defect_mean_iou:.5f}"
)

print(
    f"Defect Mean Dice:      "
    f"{defect_mean_dice:.5f}"
)

print(
    f"Defect Precision:      "
    f"{defect_mean_precision:.5f}"
)

print(
    f"Defect Recall:         "
    f"{defect_mean_recall:.5f}"
)

print(
    f"Defect Pixel Retention:"
    f" {retention:.2f}%"
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "baseline_diagnostic_metrics.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "class_id",
        "class_name",
        "gt_pixels",
        "predicted_pixels",
        "iou",
        "dice",
        "precision",
        "recall",
        "images_with_class",
        "images_predicted_class",
        "images_completely_missed"
    ])

    for class_id in range(NUM_CLASSES):

        writer.writerow([
            class_id,
            class_names[class_id],
            gt_pixel_count[class_id],
            pred_pixel_count[class_id],
            iou[class_id],
            dice[class_id],
            precision[class_id],
            recall[class_id],
            images_with_class[class_id],
            images_predicted_class[class_id],
            images_completely_missed[class_id]
        ])


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_path = os.path.join(
    OUTPUT_DIR,
    "baseline_confusion_matrix.csv"
)

np.savetxt(
    confusion_path,
    confusion_matrix,
    delimiter=",",
    fmt="%d"
)


normalized_path = os.path.join(
    OUTPUT_DIR,
    "baseline_normalized_confusion_matrix.csv"
)

np.savetxt(
    normalized_path,
    normalized_confusion,
    delimiter=",",
    fmt="%.6f"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

print()
print(
    f"Metrics saved to: {csv_path}"
)

print(
    f"Confusion matrix saved to: {confusion_path}"
)

print(
    f"Normalized confusion matrix saved to: "
    f"{normalized_path}"
)

print()