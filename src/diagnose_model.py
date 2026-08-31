import os
import sys
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

from dataset import SeverstalDataset
from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_IMAGE_DIR = os.path.join("data", "processed", "val")
VAL_CSV = os.path.join("data", "processed", "val.csv")

MODEL_PATH = os.path.join(
    "runs",
    "segmentation",
    "best_model.pth"
)

OUTPUT_DIR = os.path.join(
    "runs",
    "segmentation",
    "diagnostic"
)

BATCH_SIZE = 2
NUM_CLASSES = 5

# IMPORTANT:
# Compare the original training resolution against
# the newer 1024 x 256 resolution.
RESOLUTIONS = [
    ("800x128", 128, 800),
    ("1024x256", 256, 1024),
]


# ============================================================
# HELPERS
# ============================================================

def resize_batch(images, masks, height, width):
    """
    Resize images and masks.

    Images:
        bilinear interpolation

    Masks:
        nearest-neighbor interpolation

    This is done here instead of relying on transforms.py
    so that both resolutions can be compared fairly.
    """

    images = F.interpolate(
        images,
        size=(height, width),
        mode="bilinear",
        align_corners=False
    )

    masks = F.interpolate(
        masks.float(),
        size=(height, width),
        mode="nearest"
    ).long()

    return images, masks


def normalize_batch(images):
    """
    ImageNet normalization.
    """

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=images.device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=images.device
    ).view(1, 3, 1, 1)

    return (images - mean) / std


def load_model():
    """
    Load the trained model.
    """

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

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Sometimes the checkpoint itself is state_dict
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove possible "module." prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(confusion):
    """
    Calculate per-class IoU, Dice, precision and recall
    from the confusion matrix.

    Rows = ground truth
    Columns = prediction
    """

    results = []

    for cls in range(NUM_CLASSES):

        tp = confusion[cls, cls]

        gt_total = confusion[cls, :].sum()
        pred_total = confusion[:, cls].sum()

        false_negative = gt_total - tp
        false_positive = pred_total - tp

        union = tp + false_positive + false_negative

        if union > 0:
            iou = tp / union
        else:
            iou = 0.0

        dice_denominator = (
            2 * tp +
            false_positive +
            false_negative
        )

        if dice_denominator > 0:
            dice = (2 * tp) / dice_denominator
        else:
            dice = 0.0

        if pred_total > 0:
            precision = tp / pred_total
        else:
            precision = 0.0

        if gt_total > 0:
            recall = tp / gt_total
        else:
            recall = 0.0

        results.append({
            "class": cls,
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "gt_pixels": int(gt_total),
            "pred_pixels": int(pred_total),
            "tp": int(tp),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
        })

    return results


# ============================================================
# MAIN DIAGNOSTIC FUNCTION
# ============================================================

def evaluate_resolution(
    model,
    dataset,
    resolution_name,
    height,
    width
):

    print()
    print("=" * 70)
    print(f"DIAGNOSTIC: {resolution_name}")
    print("=" * 70)

    print(
        f"Resolution: {width} x {height}"
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    # --------------------------------------------------------
    # Global confusion matrix
    # --------------------------------------------------------

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Error counters
    # --------------------------------------------------------

    total_gt_defect_pixels = 0
    total_pred_defect_pixels = 0

    defect_missed_as_background = 0
    defect_wrong_class = 0

    background_false_positive = 0

    # --------------------------------------------------------
    # Probability statistics
    # --------------------------------------------------------

    gt_probability_sum = np.zeros(
        NUM_CLASSES,
        dtype=np.float64
    )

    gt_probability_count = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Worst images
    # --------------------------------------------------------

    worst_images = []

    processed = 0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with torch.no_grad():

        for batch_idx, batch in enumerate(loader):

            images, masks, image_ids = batch

            # Dataset mask shape:
            # [B, 1, H, W]

            masks = masks.squeeze(1)

            # Move original data to GPU/CPU
            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            # ------------------------------------------------
            # Resize
            # ------------------------------------------------

            images, masks_for_model = resize_batch(
                images,
                masks.unsqueeze(1),
                height,
                width
            )

            masks_for_model = masks_for_model.squeeze(1)

            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            images = normalize_batch(images)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            # ------------------------------------------------
            # Batch metrics
            # ------------------------------------------------

            for b in range(images.shape[0]):

                gt = masks_for_model[b]
                pred = predictions[b]

                probs = probabilities[b]

                gt_np = gt.cpu().numpy()
                pred_np = pred.cpu().numpy()

                # --------------------------------------------
                # Confusion matrix
                # --------------------------------------------

                for true_class in range(NUM_CLASSES):

                    true_pixels = (
                        gt_np == true_class
                    )

                    if not np.any(true_pixels):
                        continue

                    predicted_values = pred_np[
                        true_pixels
                    ]

                    counts = np.bincount(
                        predicted_values,
                        minlength=NUM_CLASSES
                    )

                    confusion[
                        true_class
                    ] += counts

                # --------------------------------------------
                # Defect masks
                # --------------------------------------------

                gt_defect = gt > 0
                pred_defect = pred > 0

                gt_defect_count = int(
                    gt_defect.sum().item()
                )

                pred_defect_count = int(
                    pred_defect.sum().item()
                )

                total_gt_defect_pixels += (
                    gt_defect_count
                )

                total_pred_defect_pixels += (
                    pred_defect_count
                )

                # --------------------------------------------
                # Missed defect pixels
                #
                # GT defect -> background prediction
                # --------------------------------------------

                missed = (
                    gt_defect &
                    (pred == 0)
                )

                missed_count = int(
                    missed.sum().item()
                )

                defect_missed_as_background += (
                    missed_count
                )

                # --------------------------------------------
                # Wrong defect class
                #
                # GT defect -> another defect class
                # --------------------------------------------

                wrong_class = (
                    gt_defect &
                    pred_defect &
                    (gt != pred)
                )

                wrong_class_count = int(
                    wrong_class.sum().item()
                )

                defect_wrong_class += (
                    wrong_class_count
                )

                # --------------------------------------------
                # False positives
                #
                # GT background -> defect prediction
                # --------------------------------------------

                false_positive = (
                    (~gt_defect) &
                    pred_defect
                )

                false_positive_count = int(
                    false_positive.sum().item()
                )

                background_false_positive += (
                    false_positive_count
                )

                # --------------------------------------------
                # Per-class GT probability
                # --------------------------------------------

                for cls in range(1, NUM_CLASSES):

                    cls_pixels = (
                        gt == cls
                    )

                    count = int(
                        cls_pixels.sum().item()
                    )

                    if count > 0:

                        cls_probs = probs[
                            cls
                        ][cls_pixels]

                        gt_probability_sum[cls] += (
                            cls_probs.sum().item()
                        )

                        gt_probability_count[cls] += (
                            count
                        )

                # --------------------------------------------
                # Per-image defect IoU
                # --------------------------------------------

                image_scores = []

                for cls in range(1, NUM_CLASSES):

                    gt_cls = gt == cls
                    pred_cls = pred == cls

                    intersection = (
                        gt_cls &
                        pred_cls
                    ).sum().item()

                    union = (
                        gt_cls |
                        pred_cls
                    ).sum().item()

                    if union > 0:

                        cls_iou = (
                            intersection /
                            union
                        )

                        image_scores.append(
                            cls_iou
                        )

                if len(image_scores) > 0:

                    image_mean_iou = float(
                        np.mean(image_scores)
                    )

                    worst_images.append({
                        "image_id": image_ids[b],
                        "mean_defect_iou":
                            image_mean_iou,
                        "gt_defect_pixels":
                            gt_defect_count,
                        "pred_defect_pixels":
                            pred_defect_count,
                        "missed_pixels":
                            missed_count,
                        "wrong_class_pixels":
                            wrong_class_count,
                        "false_positive_pixels":
                            false_positive_count
                    })

            processed += images.shape[0]

            if (
                processed % 200 == 0
                or processed == len(dataset)
            ):

                print(
                    f"Processed "
                    f"{processed}/{len(dataset)}"
                )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        confusion
    )

    # Mean IoU including background
    mean_iou = np.mean([
        x["iou"]
        for x in metrics
    ])

    # Mean Dice including background
    mean_dice = np.mean([
        x["dice"]
        for x in metrics
    ])

    # Defect-only metrics
    defect_iou = np.mean([
        x["iou"]
        for x in metrics[1:]
    ])

    defect_dice = np.mean([
        x["dice"]
        for x in metrics[1:]
    ])

    # --------------------------------------------------------
    # Defect retention
    # --------------------------------------------------------

    if total_gt_defect_pixels > 0:

        retention = (
            total_gt_defect_pixels
            - defect_missed_as_background
        ) / total_gt_defect_pixels

    else:
        retention = 0.0

    # --------------------------------------------------------
    # Probability statistics
    # --------------------------------------------------------

    mean_gt_probability = []

    for cls in range(1, NUM_CLASSES):

        if gt_probability_count[cls] > 0:

            value = (
                gt_probability_sum[cls] /
                gt_probability_count[cls]
            )

        else:
            value = 0.0

        mean_gt_probability.append(
            value
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)

    print(
        f"Mean IoU:          {mean_iou:.5f}"
    )

    print(
        f"Mean Dice:         {mean_dice:.5f}"
    )

    print(
        f"Defect Mean IoU:   {defect_iou:.5f}"
    )

    print(
        f"Defect Mean Dice:  {defect_dice:.5f}"
    )

    print()
    print("PER-CLASS RESULTS")
    print("-" * 70)

    for result in metrics:

        cls = result["class"]

        print(
            f"Class {cls} | "
            f"IoU: {result['iou']:.5f} | "
            f"Dice: {result['dice']:.5f} | "
            f"Precision: {result['precision']:.5f} | "
            f"Recall: {result['recall']:.5f}"
        )

    print()
    print("DEFECT ERROR BREAKDOWN")
    print("-" * 70)

    print(
        f"Ground-truth defect pixels: "
        f"{total_gt_defect_pixels:,}"
    )

    print(
        f"Predicted defect pixels:     "
        f"{total_pred_defect_pixels:,}"
    )

    print(
        f"Defects missed as background: "
        f"{defect_missed_as_background:,}"
    )

    print(
        f"Defects assigned wrong class:  "
        f"{defect_wrong_class:,}"
    )

    print(
        f"Background false positives:    "
        f"{background_false_positive:,}"
    )

    print(
        f"Approx. defect retention: "
        f"{retention * 100:.2f}%"
    )

    print()
    print("AVERAGE MODEL CONFIDENCE ON TRUE DEFECT PIXELS")
    print("-" * 70)

    for cls in range(1, NUM_CLASSES):

        print(
            f"Class {cls}: "
            f"{mean_gt_probability[cls - 1]:.5f}"
        )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print()
    print("CONFUSION MATRIX")
    print("-" * 70)

    print(
        "Rows = Ground Truth"
    )

    print(
        "Columns = Prediction"
    )

    print()

    print(confusion)

    # ========================================================
    # IMPORTANT CLASS-TO-CLASS CONFUSION
    # ========================================================

    print()
    print("DEFECT-TO-DEFECT CONFUSION")
    print("-" * 70)

    print(
        "This excludes background."
    )

    print()

    for true_cls in range(1, NUM_CLASSES):

        total = confusion[
            true_cls
        ].sum()

        if total == 0:
            continue

        print(
            f"Ground Truth Class {true_cls}:"
        )

        for pred_cls in range(NUM_CLASSES):

            count = confusion[
                true_cls,
                pred_cls
            ]

            percentage = (
                100.0 * count / total
            )

            if count > 0:

                print(
                    f"    -> Class {pred_cls}: "
                    f"{count:,} "
                    f"({percentage:.2f}%)"
                )

    # ========================================================
    # WORST IMAGES
    # ========================================================

    worst_images.sort(
        key=lambda x:
            x["mean_defect_iou"]
    )

    worst_20 = worst_images[:20]

    print()
    print("20 WORST IMAGES")
    print("-" * 70)

    for i, item in enumerate(
        worst_20,
        start=1
    ):

        print(
            f"{i:02d}. "
            f"{item['image_id']} | "
            f"IoU: "
            f"{item['mean_defect_iou']:.5f} | "
            f"GT: "
            f"{item['gt_defect_pixels']:,} | "
            f"Pred: "
            f"{item['pred_defect_pixels']:,} | "
            f"Missed: "
            f"{item['missed_pixels']:,} | "
            f"Wrong class: "
            f"{item['wrong_class_pixels']:,} | "
            f"FP: "
            f"{item['false_positive_pixels']:,}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    prefix = resolution_name

    # --------------------------------------------------------
    # Per-class CSV
    # --------------------------------------------------------

    metrics_path = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_metrics.csv"
    )

    with open(
        metrics_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "class",
            "iou",
            "dice",
            "precision",
            "recall",
            "gt_pixels",
            "pred_pixels",
            "tp",
            "false_positive",
            "false_negative"
        ])

        for result in metrics:

            writer.writerow([
                result["class"],
                result["iou"],
                result["dice"],
                result["precision"],
                result["recall"],
                result["gt_pixels"],
                result["pred_pixels"],
                result["tp"],
                result["false_positive"],
                result["false_negative"]
            ])

    # --------------------------------------------------------
    # Confusion matrix CSV
    # --------------------------------------------------------

    confusion_path = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_confusion_matrix.csv"
    )

    with open(
        confusion_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "GT\\Pred",
            0,
            1,
            2,
            3,
            4
        ])

        for cls in range(NUM_CLASSES):

            writer.writerow([
                cls,
                *confusion[cls].tolist()
            ])

    # --------------------------------------------------------
    # Worst images CSV
    # --------------------------------------------------------

    worst_path = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_worst_images.csv"
    )

    with open(
        worst_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "image_id",
            "mean_defect_iou",
            "gt_defect_pixels",
            "pred_defect_pixels",
            "missed_pixels",
            "wrong_class_pixels",
            "false_positive_pixels"
        ])

        for item in worst_images:

            writer.writerow([
                item["image_id"],
                item["mean_defect_iou"],
                item["gt_defect_pixels"],
                item["pred_defect_pixels"],
                item["missed_pixels"],
                item["wrong_class_pixels"],
                item["false_positive_pixels"]
            ])

    print()
    print("-" * 70)
    print("FILES SAVED")
    print("-" * 70)

    print(metrics_path)
    print(confusion_path)
    print(worst_path)

    return {
        "resolution": resolution_name,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "defect_iou": defect_iou,
        "defect_dice": defect_dice,
        "retention": retention,
        "confusion": confusion,
        "metrics": metrics
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STEEL DEFECT MODEL DIAGNOSTIC")
    print("=" * 70)

    print()
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()
    print(
        "This diagnostic DOES NOT train the model."
    )

    print(
        "It evaluates the existing best_model.pth "
        "at two resolutions."
    )

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not os.path.exists(
        VAL_IMAGE_DIR
    ):

        raise FileNotFoundError(
            f"Validation image directory not found:\n"
            f"{VAL_IMAGE_DIR}"
        )

    if not os.path.exists(
        VAL_CSV
    ):

        raise FileNotFoundError(
            f"Validation CSV not found:\n"
            f"{VAL_CSV}"
        )

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Dataset WITHOUT transform
    #
    # We resize manually so that the two resolutions
    # are genuinely comparable.
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("Loading validation dataset")
    print("-" * 70)

    dataset = SeverstalDataset(
        image_dir=VAL_IMAGE_DIR,
        csv_path=VAL_CSV,
        transform=None
    )

    print(
        f"Validation dataset size: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # Load model once
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Run both resolutions
    # --------------------------------------------------------

    all_results = []

    for resolution_name, height, width in RESOLUTIONS:

        result = evaluate_resolution(
            model=model,
            dataset=dataset,
            resolution_name=resolution_name,
            height=height,
            width=width
        )

        all_results.append(result)

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print("=" * 70)
    print("RESOLUTION COMPARISON")
    print("=" * 70)

    print()

    print(
        f"{'Resolution':<15}"
        f"{'Mean IoU':<15}"
        f"{'Defect IoU':<15}"
        f"{'Mean Dice':<15}"
        f"{'Defect Dice':<15}"
    )

    print("-" * 75)

    for result in all_results:

        print(
            f"{result['resolution']:<15}"
            f"{result['mean_iou']:<15.5f}"
            f"{result['defect_iou']:<15.5f}"
            f"{result['mean_dice']:<15.5f}"
            f"{result['defect_dice']:<15.5f}"
        )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    if len(all_results) == 2:

        original = all_results[0]
        higher = all_results[1]

        iou_difference = (
            higher["mean_iou"]
            - original["mean_iou"]
        )

        defect_difference = (
            higher["defect_iou"]
            - original["defect_iou"]
        )

        print()
        print("-" * 70)

        print(
            f"Mean IoU difference: "
            f"{iou_difference:+.5f}"
        )

        print(
            f"Defect IoU difference: "
            f"{defect_difference:+.5f}"
        )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)