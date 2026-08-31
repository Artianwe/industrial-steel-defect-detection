import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# Import the exact model used for the verified checkpoint.
# This script assumes model.py is in the same src/ directory.
from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_HEIGHT = 256
TARGET_WIDTH = 1024

NUM_CLASSES = 5

CLASS_NAMES = {
    0: "Background",
    1: "Defect Class 1",
    2: "Defect Class 2",
    3: "Defect Class 3",
    4: "Defect Class 4",
}

# Your verified best checkpoint.
CHECKPOINT_PATH = Path(
    "runs/segmentation_sampling/best_model.pth"
)

OUTPUT_DIR = Path("runs/predictions")


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Apply the same deterministic image preprocessing used
    during validation/training:

    1. Convert RGB image to tensor in [0, 1]
    2. Resize to 256 x 1024 using bilinear interpolation
    3. Normalize using ImageNet mean/std
    """

    image = image.convert("RGB")

    original_width, original_height = image.size

    image_array = np.asarray(image).copy()

    image_tensor = torch.from_numpy(
        image_array
    ).permute(2, 0, 1).float() / 255.0

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = F.interpolate(
        image_tensor,
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        mode="bilinear",
        align_corners=False
    )

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=image_tensor.dtype
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=image_tensor.dtype
    ).view(1, 3, 1, 1)

    image_tensor = (image_tensor - mean) / std

    return (
        image_tensor,
        (original_width, original_height)
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_model(device):
    print("=" * 70)
    print("STEEL DEFECT DETECTION")
    print("=" * 70)

    print()
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print()
    print("Loading model...")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH.resolve()}\n\n"
            "Run this script from the project root directory."
        )

    # The verified checkpoint was trained with the baseline
    # ResNet18 U-Net architecture.
    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    # Training checkpoints contain model_state_dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_epoch = checkpoint.get("epoch", "unknown")
        checkpoint_miou = checkpoint.get("best_miou", "unknown")

        print("Checkpoint epoch:", checkpoint_epoch)
        print("Checkpoint best_miou:", checkpoint_miou)

    else:
        # Also support a raw state_dict if one is supplied later.
        state_dict = checkpoint
        print("Checkpoint format: raw state_dict")

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        prediction = torch.argmax(
            logits,
            dim=1
        )

    return prediction.squeeze(0).cpu().numpy()


# ============================================================
# CREATE MASK
# ============================================================

def create_mask_image(prediction):
    """
    Create a simple RGB class mask.

    The exact colors are only for visualization.
    They do NOT change the model prediction.
    """

    # Background = black
    # Class 1 = red
    # Class 2 = green
    # Class 3 = blue
    # Class 4 = yellow
    palette = np.array([
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
    ], dtype=np.uint8)

    mask_rgb = palette[prediction]

    return Image.fromarray(mask_rgb)


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(original_image, prediction):
    """
    Overlay predicted defect regions on the original image.

    The prediction is resized back to the original image
    resolution using nearest-neighbor interpolation so that
    class labels remain discrete.
    """

    original_width, original_height = original_image.size

    prediction_tensor = torch.from_numpy(
        prediction
    ).unsqueeze(0).unsqueeze(0).float()

    prediction_tensor = F.interpolate(
        prediction_tensor,
        size=(original_height, original_width),
        mode="nearest"
    )

    prediction_original = (
        prediction_tensor
        .squeeze()
        .long()
        .numpy()
    )

    palette = np.array([
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
    ], dtype=np.uint8)

    color_mask = palette[prediction_original]

    original_array = np.asarray(
        original_image
    ).astype(np.float32)

    overlay_array = original_array.copy()

    defect_pixels = prediction_original != 0

    # Blend only predicted defect pixels.
    overlay_array[defect_pixels] = (
        0.55 * original_array[defect_pixels]
        + 0.45 * color_mask[defect_pixels]
    )

    overlay_array = np.clip(
        overlay_array,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(overlay_array)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    original_image,
    prediction,
    image_path
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    stem = Path(image_path).stem

    mask_path = OUTPUT_DIR / f"{stem}_mask.png"
    overlay_path = OUTPUT_DIR / f"{stem}_overlay.png"

    # Mask at model resolution.
    mask_image = create_mask_image(prediction)
    mask_image.save(mask_path)

    # Overlay at original image resolution.
    overlay = create_overlay(
        original_image,
        prediction
    )
    overlay.save(overlay_path)

    return mask_path, overlay_path


# ============================================================
# REPORT
# ============================================================

def print_report(prediction):
    print()
    print("=" * 70)
    print("DETECTION RESULT")
    print("=" * 70)

    detected_any = False

    total_pixels = prediction.size

    print()

    for class_id in range(1, NUM_CLASSES):

        pixel_count = int(
            np.sum(prediction == class_id)
        )

        percentage = (
            100.0 * pixel_count / total_pixels
        )

        if pixel_count > 0:
            detected = "DETECTED"
            detected_any = True
        else:
            detected = "NOT DETECTED"

        print(
            f"{CLASS_NAMES[class_id]:<20} | "
            f"{detected:<12} | "
            f"Pixels: {pixel_count:<8} | "
            f"{percentage:.4f}%"
        )

    print()

    if detected_any:
        print("Overall result: DEFECT DETECTED")
    else:
        print("Overall result: NO DEFECT DETECTED")

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Industrial steel defect segmentation inference"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the steel surface image"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path.resolve()}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Load image.
    print()
    print("Input image:", image_path)

    original_image = Image.open(
        image_path
    ).convert("RGB")

    print(
        "Original image size:",
        original_image.size
    )

    # Preprocess.
    image_tensor, original_size = preprocess_image(
        original_image
    )

    print(
        "Model input size:",
        tuple(image_tensor.shape)
    )

    # Load model.
    model = load_model(device)

    # Predict.
    print()
    print("Running inference...")

    prediction = predict(
        model,
        image_tensor,
        device
    )

    print("Inference complete.")

    # Report.
    print_report(prediction)

    # Save visualization.
    mask_path, overlay_path = save_results(
        original_image,
        prediction,
        image_path
    )

    print()
    print("Saved results:")
    print("Mask:    ", mask_path.resolve())
    print("Overlay: ", overlay_path.resolve())
    print()
    print("Open the *_overlay.png file to see the detected")
    print("defect regions on the original steel image.")


if __name__ == "__main__":
    main()
