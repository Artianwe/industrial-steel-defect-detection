import os
import sys
import time

# Make sure Python can find modules inside src/
sys.path.insert(
    0,
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import SeverstalDataset
from transforms import preprocess_pair
from model import create_model
from losses import SteelDefectLoss


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5

TARGET_HEIGHT = 256
TARGET_WIDTH = 1024

BATCH_SIZE = 2

NUM_EPOCHS = 30

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

PATIENCE = 5

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

TRAIN_IMAGE_DIR = "data/processed/train"
TRAIN_CSV = "data/processed/train.csv"

VAL_IMAGE_DIR = "data/processed/val"
VAL_CSV = "data/processed/val.csv"

# ------------------------------------------------------------
# IMPORTANT
#
# This experiment gets its own directory.
#
# VERIFIED BASELINE:
# runs/segmentation/
#
# NEW EXPERIMENT:
# runs/segmentation_sampling/
# ------------------------------------------------------------

RUN_DIR = "runs/segmentation_sampling"

BEST_MODEL_PATH = os.path.join(
    RUN_DIR,
    "best_model.pth"
)

LAST_MODEL_PATH = os.path.join(
    RUN_DIR,
    "last_model.pth"
)


# ============================================================
# CREATE RUN DIRECTORY
# ============================================================

os.makedirs(
    RUN_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DEFECT-AWARE SAMPLING EXPERIMENT")
print("=" * 70)

print()
print(f"Device: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD TRAINING DATASET
# ============================================================

print()
print("-" * 70)
print("Loading training dataset")
print("-" * 70)

train_dataset = SeverstalDataset(
    image_dir=TRAIN_IMAGE_DIR,
    csv_path=TRAIN_CSV,
    transform=preprocess_pair
)

print(
    f"Training dataset size: "
    f"{len(train_dataset)} images"
)


# ============================================================
# DEFECT-AWARE SAMPLING
# ============================================================

print()
print("-" * 70)
print("Creating defect-aware sampler")
print("-" * 70)

sampling_weights = (
    train_dataset.get_sampling_weights()
)

sampling_statistics = (
    train_dataset.get_sampling_statistics()
)

print(
    f"Minimum sampling weight: "
    f"{sampling_statistics['min_weight']:.4f}"
)

print(
    f"Maximum sampling weight: "
    f"{sampling_statistics['max_weight']:.4f}"
)

print(
    f"Mean sampling weight: "
    f"{sampling_statistics['mean_weight']:.4f}"
)

print(
    f"Number of sampled images: "
    f"{sampling_statistics['num_images']}"
)


train_sampler = WeightedRandomSampler(
    weights=sampling_weights,
    num_samples=len(train_dataset),
    replacement=True
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print()
print("-" * 70)
print("Loading validation dataset")
print("-" * 70)

val_dataset = SeverstalDataset(
    image_dir=VAL_IMAGE_DIR,
    csv_path=VAL_CSV,
    transform=preprocess_pair
)

print(
    f"Validation dataset size: "
    f"{len(val_dataset)} images"
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print()
print(
    f"Training batches per epoch: "
    f"{len(train_loader)}"
)

print(
    f"Validation batches: "
    f"{len(val_loader)}"
)


# ============================================================
# MODEL
# ============================================================

print()
print("-" * 70)
print("Creating model")
print("-" * 70)

model = create_model(
    num_classes=NUM_CLASSES,
    pretrained=True
)

model = model.to(device)

print(
    f"Model parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# LOSS
# ============================================================

print()
print("-" * 70)
print("Creating loss function")
print("-" * 70)

criterion = SteelDefectLoss(
    num_classes=NUM_CLASSES
)

criterion = criterion.to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    predictions,
    targets,
    num_classes=5
):

    ious = []
    dices = []

    for class_id in range(num_classes):

        pred_mask = (
            predictions == class_id
        )

        target_mask = (
            targets == class_id
        )

        intersection = (
            pred_mask & target_mask
        ).sum().item()

        pred_pixels = (
            pred_mask.sum().item()
        )

        target_pixels = (
            target_mask.sum().item()
        )

        union = (
            pred_pixels
            +
            target_pixels
            -
            intersection
        )

        if union > 0:

            iou = (
                intersection / union
            )

        else:

            iou = 0.0

        dice_denominator = (
            pred_pixels
            +
            target_pixels
        )

        if dice_denominator > 0:

            dice = (
                2.0
                *
                intersection
                /
                dice_denominator
            )

        else:

            dice = 0.0

        ious.append(iou)
        dices.append(dice)

    mean_iou = sum(ious) / num_classes

    mean_dice = sum(dices) / num_classes

    defect_mean_iou = (
        sum(ious[1:])
        /
        (num_classes - 1)
    )

    defect_mean_dice = (
        sum(dices[1:])
        /
        (num_classes - 1)
    )

    return (
        ious,
        dices,
        mean_iou,
        mean_dice,
        defect_mean_iou,
        defect_mean_dice
    )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_dice_loss = 0.0

    num_batches = 0

    for images, masks, _ in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        if masks.ndim == 4:

            masks = masks.squeeze(1)

        masks = masks.long()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(images)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss_output = criterion(
            outputs,
            masks
        )

        # ----------------------------------------------------
        # Support both possible loss return styles
        # ----------------------------------------------------

        if isinstance(
            loss_output,
            tuple
        ):

            loss = loss_output[0]

            if len(loss_output) > 1:

                ce_loss = loss_output[1]

            else:

                ce_loss = loss.detach()

            if len(loss_output) > 2:

                dice_loss = loss_output[2]

            else:

                dice_loss = loss.detach()

        elif isinstance(
            loss_output,
            dict
        ):

            loss = loss_output.get(
                "total_loss",
                loss_output.get(
                    "loss"
                )
            )

            ce_loss = loss_output.get(
                "ce_loss",
                loss.detach()
            )

            dice_loss = loss_output.get(
                "dice_loss",
                loss.detach()
            )

        else:

            loss = loss_output

            ce_loss = loss.detach()

            dice_loss = loss.detach()

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Accumulate losses
        # ----------------------------------------------------

        total_loss += loss.item()

        total_ce_loss += (
            ce_loss.item()
        )

        total_dice_loss += (
            dice_loss.item()
        )

        num_batches += 1

    return (
        total_loss / num_batches,
        total_ce_loss / num_batches,
        total_dice_loss / num_batches
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    total_batches = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for images, masks, _ in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            if masks.ndim == 4:

                masks = masks.squeeze(1)

            masks = masks.long()

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            outputs = model(images)

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss_output = criterion(
                outputs,
                masks
            )

            if isinstance(
                loss_output,
                tuple
            ):

                loss = loss_output[0]

            elif isinstance(
                loss_output,
                dict
            ):

                loss = loss_output.get(
                    "total_loss",
                    loss_output.get(
                        "loss"
                    )
                )

            else:

                loss = loss_output

            total_loss += loss.item()

            total_batches += 1

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.append(
                predictions.cpu()
            )

            all_targets.append(
                masks.cpu()
            )

    all_predictions = torch.cat(
        all_predictions,
        dim=0
    )

    all_targets = torch.cat(
        all_targets,
        dim=0
    )

    (
        ious,
        dices,
        mean_iou,
        mean_dice,
        defect_mean_iou,
        defect_mean_dice
    ) = calculate_metrics(
        all_predictions,
        all_targets,
        NUM_CLASSES
    )

    average_loss = (
        total_loss
        /
        total_batches
    )

    return (
        average_loss,
        ious,
        dices,
        mean_iou,
        mean_dice,
        defect_mean_iou,
        defect_mean_dice
    )


# ============================================================
# TRAINING LOOP
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

best_defect_miou = 0.0

best_mean_iou = 0.0

patience_counter = 0

start_epoch = 1


for epoch in range(
    start_epoch,
    NUM_EPOCHS + 1
):

    epoch_start_time = time.time()

    print()
    print("=" * 70)

    print(
        f"EPOCH {epoch}/{NUM_EPOCHS}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    (
        train_loss,
        train_ce_loss,
        train_dice_loss
    ) = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    (
        val_loss,
        ious,
        dices,
        mean_iou,
        mean_dice,
        defect_mean_iou,
        defect_mean_dice
    ) = validate(
        model,
        val_loader,
        criterion,
        device
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(
        defect_mean_iou
    )

    current_lr = optimizer.param_groups[0][
        "lr"
    ]

    # --------------------------------------------------------
    # Epoch time
    # --------------------------------------------------------

    epoch_time = (
        time.time()
        -
        epoch_start_time
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("EPOCH RESULTS")
    print("-" * 70)

    print(
        f"Train Loss:          "
        f"{train_loss:.5f}"
    )

    print(
        f"Train CE Loss:       "
        f"{train_ce_loss:.5f}"
    )

    print(
        f"Train Dice Loss:     "
        f"{train_dice_loss:.5f}"
    )

    print(
        f"Val Loss:            "
        f"{val_loss:.5f}"
    )

    print(
        f"Mean IoU:            "
        f"{mean_iou:.5f}"
    )

    print(
        f"Mean Dice:           "
        f"{mean_dice:.5f}"
    )

    print(
        f"Defect Mean IoU:     "
        f"{defect_mean_iou:.5f}"
    )

    print(
        f"Defect Mean Dice:    "
        f"{defect_mean_dice:.5f}"
    )

    print()

    for class_id in range(
        NUM_CLASSES
    ):

        print(
            f"Class {class_id} | "
            f"IoU: {ious[class_id]:.5f} | "
            f"Dice: {dices[class_id]:.5f}"
        )

    print()

    print(
        f"Learning rate: "
        f"{current_lr:.8f}"
    )

    print(
        f"Epoch time: "
        f"{epoch_time / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    if (
        defect_mean_iou
        >
        best_defect_miou
    ):

        best_defect_miou = (
            defect_mean_iou
        )

        best_mean_iou = (
            mean_iou
        )

        patience_counter = 0

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "best_miou":
                best_defect_miou,

            "best_mean_iou":
                best_mean_iou,

            "experiment":
                "defect_aware_sampling"
        }

        torch.save(
            checkpoint,
            BEST_MODEL_PATH
        )

        print()
        print("NEW BEST MODEL!")

        print(
            f"Best Defect Mean IoU: "
            f"{best_defect_miou:.5f}"
        )

        print(
            f"Corresponding Mean IoU: "
            f"{best_mean_iou:.5f}"
        )

    else:

        patience_counter += 1

        print()
        print("No improvement.")

        print(
            f"Patience: "
            f"{patience_counter}/{PATIENCE}"
        )

    # --------------------------------------------------------
    # Save latest checkpoint
    # --------------------------------------------------------

    last_checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_miou":
            best_defect_miou,

        "best_mean_iou":
            best_mean_iou,

        "experiment":
            "defect_aware_sampling"
    }

    torch.save(
        last_checkpoint,
        LAST_MODEL_PATH
    )

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            /
            1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            /
            1024**3
        )

        peak = (
            torch.cuda.max_memory_allocated()
            /
            1024**3
        )

        print()
        print("GPU MEMORY")
        print("-" * 70)

        print(
            f"Allocated: {allocated:.2f} GB"
        )

        print(
            f"Reserved:  {reserved:.2f} GB"
        )

        print(
            f"Peak:      {peak:.2f} GB"
        )

        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        patience_counter
        >=
        PATIENCE
    ):

        print()
        print("=" * 70)
        print("EARLY STOPPING")
        print("=" * 70)

        print(
            f"No improvement for "
            f"{PATIENCE} epochs."
        )

        break


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()

print(
    f"Best Defect Mean IoU: "
    f"{best_defect_miou:.5f}"
)

print(
    f"Corresponding Mean IoU: "
    f"{best_mean_iou:.5f}"
)

print()

print(
    f"Best model saved to:"
)

print(
    BEST_MODEL_PATH
)

print()

print(
    f"Last model saved to:"
)

print(
    LAST_MODEL_PATH
)

print()
print("=" * 70)