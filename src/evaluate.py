import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SeverstalDataset
from transforms import preprocess_pair
from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = "data/processed/val"
CSV_PATH = "data/processed/val.csv"
MODEL_PATH = "runs/segmentation/best_model.pth"

NUM_CLASSES = 5
BATCH_SIZE = 2
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_confusion_matrix(pred, target, num_classes):
    """
    Calculate confusion matrix.

    Rows    = ground truth
    Columns = prediction
    """

    pred = pred.reshape(-1)
    target = target.reshape(-1)

    valid = (
        (target >= 0)
        & (target < num_classes)
    )

    target = target[valid]
    pred = pred[valid]

    confusion = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
        device=pred.device
    )

    indices = (
        target * num_classes + pred
    )

    confusion += torch.bincount(
        indices,
        minlength=num_classes * num_classes
    ).reshape(
        num_classes,
        num_classes
    )

    return confusion


def calculate_metrics(confusion):
    """
    Calculate IoU and Dice for every class.
    """

    confusion = confusion.float()

    true_positive = torch.diag(confusion)

    false_positive = (
        confusion.sum(dim=0)
        - true_positive
    )

    false_negative = (
        confusion.sum(dim=1)
        - true_positive
    )

    union = (
        true_positive
        + false_positive
        + false_negative
    )

    iou = torch.zeros_like(true_positive)

    valid_iou = union > 0

    iou[valid_iou] = (
        true_positive[valid_iou]
        / union[valid_iou]
    )

    dice_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    dice = torch.zeros_like(true_positive)

    valid_dice = dice_denominator > 0

    dice[valid_dice] = (
        2 * true_positive[valid_dice]
        / dice_denominator[valid_dice]
    )

    return iou, dice


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEEL DEFECT SEGMENTATION EVALUATION")
    print("=" * 70)

    print()
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print()
    print(f"Model: {MODEL_PATH}")
    print(f"Validation images: {IMAGE_DIR}")
    print(f"Validation CSV: {CSV_PATH}")

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(IMAGE_DIR):

        raise FileNotFoundError(
            f"Validation image directory not found:\n{IMAGE_DIR}"
        )

    if not os.path.exists(CSV_PATH):

        raise FileNotFoundError(
            f"Validation CSV not found:\n{CSV_PATH}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("Loading validation dataset")
    print("-" * 70)

    dataset = SeverstalDataset(
        image_dir=IMAGE_DIR,
        csv_path=CSV_PATH,
        transform=preprocess_pair
    )

    print(
        f"Validation dataset size: {len(dataset)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Number of batches: {len(loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("Loading model")
    print("-" * 70)

    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Handle both:
    # 1. plain state_dict
    # 2. checkpoint dictionary

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Remove possible DataParallel prefix

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model = model.to(DEVICE)

    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("Running validation evaluation")
    print("-" * 70)

    confusion_matrix = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=torch.int64,
        device=DEVICE
    )

    total_images = 0

    # Statistics for missed defects

    images_with_defects = 0
    images_with_missed_defects = 0
    total_ground_truth_defect_pixels = 0
    total_predicted_defect_pixels = 0
    total_missed_defect_pixels = 0

    with torch.no_grad():

        for batch_idx, batch in enumerate(loader):

            images, masks, image_ids = batch

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            # Mask shape:
            # [B, 1, H, W]

            masks = masks.squeeze(1)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(images)

            # ------------------------------------------------
            # Convert logits to class predictions
            # ------------------------------------------------

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            # ------------------------------------------------
            # Confusion matrix
            # ------------------------------------------------

            confusion_matrix += (
                calculate_confusion_matrix(
                    predictions,
                    masks,
                    NUM_CLASSES
                )
            )

            # ------------------------------------------------
            # Defect statistics
            # ------------------------------------------------

            for i in range(len(image_ids)):

                gt = masks[i]
                pred = predictions[i]

                gt_defect = gt > 0
                pred_defect = pred > 0

                gt_pixels = (
                    gt_defect.sum().item()
                )

                pred_pixels = (
                    pred_defect.sum().item()
                )

                missed_pixels = (
                    gt_defect
                    & (~pred_defect)
                ).sum().item()

                total_ground_truth_defect_pixels += (
                    gt_pixels
                )

                total_predicted_defect_pixels += (
                    pred_pixels
                )

                total_missed_defect_pixels += (
                    missed_pixels
                )

                if gt_pixels > 0:

                    images_with_defects += 1

                    if missed_pixels > 0:

                        images_with_missed_defects += 1

            total_images += len(image_ids)

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                (batch_idx + 1) % 100 == 0
                or batch_idx == 0
                or batch_idx == len(loader) - 1
            ):

                print(
                    f"Processed "
                    f"{total_images}/{len(dataset)} "
                    f"images"
                )

    # ========================================================
    # METRICS
    # ========================================================

    iou, dice = calculate_metrics(
        confusion_matrix
    )

    iou = iou.cpu().numpy()
    dice = dice.cpu().numpy()

    # --------------------------------------------------------
    # Mean metrics
    # --------------------------------------------------------

    # Include all 5 classes because this is how
    # the training script reported Mean IoU.

    mean_iou = float(
        np.mean(iou)
    )

    mean_dice = float(
        np.mean(dice)
    )

    # Defect-only metrics

    defect_mean_iou = float(
        np.mean(iou[1:])
    )

    defect_mean_dice = float(
        np.mean(dice[1:])
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print()
    print(
        f"Mean IoU:       {mean_iou:.5f}"
    )

    print(
        f"Mean Dice:      {mean_dice:.5f}"
    )

    print()
    print(
        f"Defect Mean IoU:  {defect_mean_iou:.5f}"
    )

    print(
        f"Defect Mean Dice: {defect_mean_dice:.5f}"
    )

    print()
    print("-" * 70)
    print("PER-CLASS RESULTS")
    print("-" * 70)

    for class_id in range(NUM_CLASSES):

        print(
            f"Class {class_id} | "
            f"IoU: {iou[class_id]:.5f} | "
            f"Dice: {dice[class_id]:.5f}"
        )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print()
    print("-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    print(
        "Rows = Ground Truth"
    )

    print(
        "Columns = Prediction"
    )

    print()

    confusion_cpu = (
        confusion_matrix
        .cpu()
        .numpy()
    )

    print(
        confusion_cpu
    )

    # ========================================================
    # DEFECT STATISTICS
    # ========================================================

    print()
    print("-" * 70)
    print("DEFECT DETECTION STATISTICS")
    print("-" * 70)

    print(
        f"Images evaluated: "
        f"{total_images}"
    )

    print(
        f"Images containing defects: "
        f"{images_with_defects}"
    )

    print(
        f"Images with missed defect pixels: "
        f"{images_with_missed_defects}"
    )

    print(
        f"Ground-truth defect pixels: "
        f"{total_ground_truth_defect_pixels:,}"
    )

    print(
        f"Predicted defect pixels: "
        f"{total_predicted_defect_pixels:,}"
    )

    print(
        f"Missed defect pixels: "
        f"{total_missed_defect_pixels:,}"
    )

    if total_ground_truth_defect_pixels > 0:

        retention = (
            1.0
            - (
                total_missed_defect_pixels
                / total_ground_truth_defect_pixels
            )
        ) * 100.0

        print(
            f"Approx. defect pixel retention: "
            f"{retention:.2f}%"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_dir = "runs/segmentation/evaluation"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    metrics_path = os.path.join(
        output_dir,
        "evaluation_metrics.csv"
    )

    with open(
        metrics_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "ClassId",
                "IoU",
                "Dice"
            ]
        )

        for class_id in range(NUM_CLASSES):

            writer.writerow(
                [
                    class_id,
                    f"{iou[class_id]:.6f}",
                    f"{dice[class_id]:.6f}"
                ]
            )

        writer.writerow(
            [
                "Mean",
                f"{mean_iou:.6f}",
                f"{mean_dice:.6f}"
            ]
        )

        writer.writerow(
            [
                "DefectOnlyMean",
                f"{defect_mean_iou:.6f}",
                f"{defect_mean_dice:.6f}"
            ]
        )

    confusion_path = os.path.join(
        output_dir,
        "confusion_matrix.csv"
    )

    np.savetxt(
        confusion_path,
        confusion_cpu,
        delimiter=",",
        fmt="%d"
    )

    print()
    print("-" * 70)
    print("FILES SAVED")
    print("-" * 70)

    print(
        f"Metrics: {metrics_path}"
    )

    print(
        f"Confusion matrix: {confusion_path}"
    )

    # ========================================================
    # FINAL CHECK
    # ========================================================

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()