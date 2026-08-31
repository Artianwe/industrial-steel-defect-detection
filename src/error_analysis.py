import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from dataset import SeverstalDataset
from transforms import preprocess_pair
from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = "data/processed/val"
CSV_PATH = "data/processed/val.csv"
MODEL_PATH = "runs/segmentation/best_model.pth"

OUTPUT_DIR = "runs/segmentation/error_analysis"

NUM_CLASSES = 5
BATCH_SIZE = 1
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TOP_N = 10


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

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

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

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

    return model


# ============================================================
# CALCULATE CLASS IOU
# ============================================================

def calculate_class_iou(
    ground_truth,
    prediction,
    class_id
):

    gt = ground_truth == class_id
    pred = prediction == class_id

    intersection = np.logical_and(
        gt,
        pred
    ).sum()

    union = np.logical_or(
        gt,
        pred
    ).sum()

    if union == 0:
        return None

    return float(
        intersection / union
    )


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_visualization(
    image_tensor,
    ground_truth,
    prediction,
    output_path
):

    # --------------------------------------------------------
    # Undo ImageNet normalization
    # --------------------------------------------------------

    image = image_tensor.cpu().numpy()

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    ).reshape(3, 1, 1)

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    ).reshape(3, 1, 1)

    image = image * std + mean

    image = np.clip(
        image,
        0,
        1
    )

    image = (
        image.transpose(1, 2, 0) * 255
    ).astype(np.uint8)

    ground_truth = ground_truth.cpu().numpy()
    prediction = prediction.cpu().numpy()

    # --------------------------------------------------------
    # Mask colors
    # --------------------------------------------------------

    gt_rgb = np.zeros(
        (*ground_truth.shape, 3),
        dtype=np.uint8
    )

    pred_rgb = np.zeros(
        (*prediction.shape, 3),
        dtype=np.uint8
    )

    colors = {
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (255, 255, 0)
    }

    for class_id, color in colors.items():

        gt_rgb[
            ground_truth == class_id
        ] = color

        pred_rgb[
            prediction == class_id
        ] = color

    # --------------------------------------------------------
    # False negatives
    # --------------------------------------------------------

    false_negative = (
        (ground_truth > 0)
        & (prediction == 0)
    )

    fn_rgb = np.zeros_like(gt_rgb)

    fn_rgb[
        false_negative
    ] = (255, 0, 0)

    # --------------------------------------------------------
    # False positives
    # --------------------------------------------------------

    false_positive = (
        (ground_truth == 0)
        & (prediction > 0)
    )

    fp_rgb = np.zeros_like(gt_rgb)

    fp_rgb[
        false_positive
    ] = (0, 255, 0)

    # --------------------------------------------------------
    # Overlays
    # --------------------------------------------------------

    gt_overlay = (
        0.65 * image
        + 0.35 * gt_rgb
    ).clip(0, 255).astype(np.uint8)

    pred_overlay = (
        0.65 * image
        + 0.35 * pred_rgb
    ).clip(0, 255).astype(np.uint8)

    fn_overlay = (
        0.65 * image
        + 0.35 * fn_rgb
    ).clip(0, 255).astype(np.uint8)

    fp_overlay = (
        0.65 * image
        + 0.35 * fp_rgb
    ).clip(0, 255).astype(np.uint8)

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = np.concatenate(
        [
            image,
            gt_overlay,
            pred_overlay,
            fn_overlay,
            fp_overlay
        ],
        axis=1
    )

    Image.fromarray(
        combined
    ).save(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEEL DEFECT ERROR ANALYSIS")
    print("=" * 70)

    print()
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    worst_dir = os.path.join(
        OUTPUT_DIR,
        "worst_overall"
    )

    os.makedirs(
        worst_dir,
        exist_ok=True
    )

    class_dirs = {}

    for class_id in range(1, NUM_CLASSES):

        class_dirs[class_id] = os.path.join(
            OUTPUT_DIR,
            f"class_{class_id}"
        )

        os.makedirs(
            class_dirs[class_id],
            exist_ok=True
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
        f"Validation images: {len(dataset)}"
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Store ONLY metrics + dataset index
    #
    # We do NOT store tensors for all images.
    # --------------------------------------------------------

    overall_results = []

    class_results = {
        1: [],
        2: [],
        3: [],
        4: []
    }

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("Analyzing validation images")
    print("-" * 70)

    for index, batch in enumerate(loader):

        try:

            images, masks, image_ids = batch

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.squeeze(1)

            with torch.inference_mode():

                outputs = model(images)

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

            gt = masks[0].cpu()
            pred = predictions[0].cpu()
            image = images[0].cpu()

            gt_np = gt.numpy()
            pred_np = pred.numpy()

            image_id = image_ids[0]

            # ------------------------------------------------
            # Overall defect IoU
            # ------------------------------------------------

            gt_defect = gt_np > 0
            pred_defect = pred_np > 0

            intersection = np.logical_and(
                gt_defect,
                pred_defect
            ).sum()

            union = np.logical_or(
                gt_defect,
                pred_defect
            ).sum()

            if union > 0:

                defect_iou = (
                    intersection / union
                )

            else:

                defect_iou = 1.0

            missed_pixels = np.logical_and(
                gt_defect,
                ~pred_defect
            ).sum()

            false_positive_pixels = np.logical_and(
                ~gt_defect,
                pred_defect
            ).sum()

            # ------------------------------------------------
            # Store only lightweight information
            # ------------------------------------------------

            overall_results.append(
                {
                    "index": index,
                    "image_id": image_id,
                    "iou": float(defect_iou),
                    "missed_pixels": int(
                        missed_pixels
                    ),
                    "false_positive_pixels": int(
                        false_positive_pixels
                    )
                }
            )

            # ------------------------------------------------
            # Class-specific IoU
            # ------------------------------------------------

            for class_id in range(
                1,
                NUM_CLASSES
            ):

                gt_pixels = (
                    gt_np == class_id
                ).sum()

                if gt_pixels > 0:

                    class_iou = calculate_class_iou(
                        gt_np,
                        pred_np,
                        class_id
                    )

                    class_results[
                        class_id
                    ].append(
                        {
                            "index": index,
                            "image_id": image_id,
                            "iou": class_iou,
                            "gt_pixels": int(
                                gt_pixels
                            )
                        }
                    )

            # ------------------------------------------------
            # Explicitly delete tensors
            # ------------------------------------------------

            del images
            del masks
            del outputs
            del predictions
            del gt
            del pred
            del image

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                (index + 1) % 50 == 0
                or index == 0
                or index == len(dataset) - 1
            ):

                print(
                    f"Processed "
                    f"{index + 1}/{len(dataset)}"
                )

        except Exception as e:

            print()
            print(
                f"ERROR at image "
                f"{index}"
            )

            print(e)

            raise

    # ========================================================
    # SORT RESULTS
    # ========================================================

    overall_results.sort(
        key=lambda x: x["iou"]
    )

    for class_id in range(1, NUM_CLASSES):

        class_results[class_id].sort(
            key=lambda x: (
                x["iou"]
                if x["iou"] is not None
                else 1.0
            )
        )

    # ========================================================
    # GENERATE WORST OVERALL
    # ========================================================

    print()
    print("=" * 70)
    print("WORST OVERALL EXAMPLES")
    print("=" * 70)

    for rank, item in enumerate(
        overall_results[:TOP_N],
        start=1
    ):

        index = item["index"]

        images, masks, image_ids = dataset[
            index
        ]

        with torch.inference_mode():

            image_input = images.unsqueeze(
                0
            ).to(DEVICE)

            output = model(
                image_input
            )

            prediction = torch.argmax(
                output,
                dim=1
            )[0].cpu()

        output_path = os.path.join(
            worst_dir,
            f"{rank:02d}_{item['image_id']}.png"
        )

        save_visualization(
            images,
            masks.squeeze(0),
            prediction,
            output_path
        )

        print(
            f"{rank:02d}. "
            f"{item['image_id']} | "
            f"IoU = {item['iou']:.4f} | "
            f"Missed = "
            f"{item['missed_pixels']:,}"
        )

        del image_input
        del output
        del prediction

    # ========================================================
    # GENERATE WORST PER CLASS
    # ========================================================

    for class_id in range(
        1,
        NUM_CLASSES
    ):

        print()
        print("=" * 70)
        print(
            f"WORST CLASS {class_id} EXAMPLES"
        )
        print("=" * 70)

        examples = class_results[
            class_id
        ]

        for rank, item in enumerate(
            examples[:TOP_N],
            start=1
        ):

            index = item["index"]

            images, masks, image_ids = dataset[
                index
            ]

            with torch.inference_mode():

                image_input = images.unsqueeze(
                    0
                ).to(DEVICE)

                output = model(
                    image_input
                )

                prediction = torch.argmax(
                    output,
                    dim=1
                )[0].cpu()

            output_path = os.path.join(
                class_dirs[class_id],
                f"{rank:02d}_{item['image_id']}.png"
            )

            save_visualization(
                images,
                masks.squeeze(0),
                prediction,
                output_path
            )

            print(
                f"{rank:02d}. "
                f"{item['image_id']} | "
                f"IoU = {item['iou']:.4f} | "
                f"GT pixels = "
                f"{item['gt_pixels']:,}"
            )

            del image_input
            del output
            del prediction

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Generated folders:"
    )

    print(
        "  worst_overall/"
    )

    for class_id in range(
        1,
        NUM_CLASSES
    ):

        print(
            f"  class_{class_id}/"
        )


if __name__ == "__main__":
    main()