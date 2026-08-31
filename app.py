import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

import torch
import torch.nn.functional as F

from src.model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_HEIGHT = 256
TARGET_WIDTH = 1024
NUM_CLASSES = 5

# Defect-area decision thresholds.
# These apply to the percentage of the full image predicted
# as a particular defect class.
MIN_DEFECT_AREA_PERCENT = 0.05
CONFIRMED_DEFECT_AREA_PERCENT = 0.50

# Minimum number of pixels required after resizing to the
# original image before a class is considered at all.
MIN_DEFECT_PIXELS = 20


# ============================================================
# SEVERSTAL DEFECT CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = {
    0: "Background",
    1: "Pitted Surface",
    2: "Crazing",
    3: "Scratches",
    4: "Patches",
}

CLASS_DESCRIPTIONS = {
    0: "Normal steel surface",
    1: "Small pits or cavities on the steel surface",
    2: "Fine crack-like network patterns",
    3: "Linear scratches or elongated surface marks",
    4: "Irregular patch-like surface defects",
}


# ============================================================
# VISUALIZATION COLORS
# ============================================================

# Class 0 = Background
# Class 1 = Yellow   -> Pitted Surface
# Class 2 = Blue     -> Crazing
# Class 3 = Purple   -> Scratches
# Class 4 = Red      -> Patches

PALETTE = np.array([
    [0, 0, 0],
    [255, 255, 0],
    [0, 100, 255],
    [160, 32, 240],
    [255, 0, 0],
], dtype=np.uint8)


# ============================================================
# MODEL CHECKPOINT
# ============================================================

CHECKPOINT_PATH = Path(
    "runs/segmentation_sampling/best_model.pth"
)

OUTPUT_DIR = Path(
    "runs/inspection_results"
)


# ============================================================
# MODEL
# ============================================================

class SteelInspectionModel:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = create_model(
            num_classes=NUM_CLASSES,
            pretrained=False
        )

        if not CHECKPOINT_PATH.exists():

            raise FileNotFoundError(
                "Checkpoint not found:\n"
                f"{CHECKPOINT_PATH.resolve()}\n\n"
                "Run this application from the project root."
            )

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()


    # ========================================================
    # PREPROCESS
    # ========================================================

    def preprocess(self, image):

        image_array = np.asarray(
            image.convert("RGB")
        ).copy()

        tensor = torch.from_numpy(
            image_array
        ).permute(2, 0, 1).float() / 255.0

        tensor = tensor.unsqueeze(0)

        tensor = F.interpolate(
            tensor,
            size=(TARGET_HEIGHT, TARGET_WIDTH),
            mode="bilinear",
            align_corners=False
        )

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=tensor.dtype
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=tensor.dtype
        ).view(1, 3, 1, 1)

        tensor = (tensor - mean) / std

        return tensor.to(self.device)


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, image):

        tensor = self.preprocess(image)

        with torch.no_grad():

            logits = self.model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            prediction = torch.argmax(
                logits,
                dim=1
            )

            confidence = torch.max(
                probabilities,
                dim=1
            ).values

        prediction = (
            prediction
            .squeeze(0)
            .cpu()
            .numpy()
        )

        confidence = (
            confidence
            .squeeze(0)
            .cpu()
            .numpy()
        )

        return prediction, confidence


# ============================================================
# RESIZE PREDICTION TO ORIGINAL IMAGE
# ============================================================

def resize_prediction_to_original(
    prediction,
    original_size
):

    width, height = original_size

    prediction_tensor = torch.from_numpy(
        prediction
    ).unsqueeze(0).unsqueeze(0).float()

    prediction_tensor = F.interpolate(
        prediction_tensor,
        size=(height, width),
        mode="nearest"
    )

    prediction_original = (
        prediction_tensor
        .squeeze()
        .long()
        .numpy()
    )

    return prediction_original


# ============================================================
# OVERLAY
# ============================================================

def make_overlay(
    original_image,
    prediction
):

    original_image = original_image.convert("RGB")

    prediction_original = resize_prediction_to_original(
        prediction,
        original_image.size
    )

    color_mask = PALETTE[
        prediction_original
    ]

    original = np.asarray(
        original_image
    ).astype(np.float32)

    overlay = original.copy()

    defect_pixels = prediction_original != 0

    overlay[defect_pixels] = (
        0.55 * original[defect_pixels]
        +
        0.45 * color_mask[defect_pixels]
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(overlay)


# ============================================================
# APPLICATION
# ============================================================

class InspectionApp:

    def __init__(
        self,
        root,
        inspector
    ):

        self.root = root
        self.inspector = inspector

        self.root.title(
            "Industrial Steel Defect Inspection"
        )

        self.root.geometry(
            "1250x900"
        )

        self.root.minsize(
            1050,
            750
        )

        self.original_image = None
        self.overlay_image = None
        self.prediction = None
        self.confidence = None
        self.last_report = ""

        self.tk_original = None
        self.tk_overlay = None

        self.build_interface()


    # ========================================================
    # USER INTERFACE
    # ========================================================

    def build_interface(self):

        title = tk.Label(
            self.root,
            text="INDUSTRIAL STEEL DEFECT INSPECTION",
            font=("Arial", 20, "bold")
        )

        title.pack(
            pady=(12, 3)
        )


        subtitle = tk.Label(
            self.root,
            text=(
                "AI-powered semantic segmentation of steel "
                "surface defects"
            ),
            font=("Arial", 10)
        )

        subtitle.pack(
            pady=(0, 3)
        )


        device_text = (
            f"AI Model: ResNet18 U-Net    |    "
            f"Device: {self.inspector.device}"
        )

        device_label = tk.Label(
            self.root,
            text=device_text,
            font=("Arial", 10)
        )

        device_label.pack(
            pady=(0, 8)
        )


        # ----------------------------------------------------
        # BUTTONS
        # ----------------------------------------------------

        button_frame = tk.Frame(
            self.root
        )

        button_frame.pack(
            pady=5
        )


        upload_button = tk.Button(
            button_frame,
            text="UPLOAD STEEL IMAGE",
            command=self.upload_image,
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10
        )

        upload_button.pack(
            side=tk.LEFT,
            padx=8
        )


        save_button = tk.Button(
            button_frame,
            text="SAVE RESULT",
            command=self.save_result,
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10
        )

        save_button.pack(
            side=tk.LEFT,
            padx=8
        )


        self.status_label = tk.Label(
            self.root,
            text="Upload an image to begin inspection.",
            font=("Arial", 12, "bold")
        )

        self.status_label.pack(
            pady=6
        )


        # ----------------------------------------------------
        # IMAGE DISPLAY
        # ----------------------------------------------------

        image_frame = tk.Frame(
            self.root
        )

        image_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=15,
            pady=5
        )


        left_frame = tk.LabelFrame(
            image_frame,
            text="ORIGINAL IMAGE",
            font=("Arial", 11, "bold")
        )

        left_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=5
        )


        right_frame = tk.LabelFrame(
            image_frame,
            text="AI DEFECT SEGMENTATION",
            font=("Arial", 11, "bold")
        )

        right_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=5
        )


        self.original_label = tk.Label(
            left_frame,
            text="No image loaded"
        )

        self.original_label.pack(
            fill=tk.BOTH,
            expand=True
        )


        self.overlay_label = tk.Label(
            right_frame,
            text="No prediction yet"
        )

        self.overlay_label.pack(
            fill=tk.BOTH,
            expand=True
        )


        # ----------------------------------------------------
        # LEGEND
        # ----------------------------------------------------

        legend_frame = tk.LabelFrame(
            self.root,
            text="DEFECT LEGEND",
            font=("Arial", 11, "bold")
        )

        legend_frame.pack(
            fill=tk.X,
            padx=20,
            pady=(5, 5)
        )


        legend_inner = tk.Frame(
            legend_frame
        )

        legend_inner.pack(
            pady=6
        )


        for class_id in range(1, NUM_CLASSES):

            color = tuple(
                int(x)
                for x in PALETTE[class_id]
            )

            color_hex = "#{:02x}{:02x}{:02x}".format(
                *color
            )

            item = tk.Frame(
                legend_inner
            )

            item.pack(
                side=tk.LEFT,
                padx=12
            )


            color_box = tk.Label(
                item,
                bg=color_hex,
                width=3,
                height=1,
                relief=tk.SOLID,
                borderwidth=1
            )

            color_box.pack(
                side=tk.LEFT,
                padx=(0, 5)
            )


            text = tk.Label(
                item,
                text=(
                    f"Class {class_id}: "
                    f"{CLASS_NAMES[class_id]}"
                ),
                font=("Arial", 10, "bold")
            )

            text.pack(
                side=tk.LEFT
            )


        # ----------------------------------------------------
        # DECISION THRESHOLD INFORMATION
        # ----------------------------------------------------

        threshold_label = tk.Label(
            self.root,
            text=(
                f"Decision rule: < {MIN_DEFECT_AREA_PERCENT:.2f}% "
                f"= ignored    |    "
                f"{MIN_DEFECT_AREA_PERCENT:.2f}%–"
                f"{CONFIRMED_DEFECT_AREA_PERCENT:.2f}% = possible    |    "
                f">= {CONFIRMED_DEFECT_AREA_PERCENT:.2f}% = confirmed"
            ),
            font=("Arial", 9)
        )

        threshold_label.pack(
            pady=(0, 3)
        )


        # ----------------------------------------------------
        # INSPECTION REPORT
        # ----------------------------------------------------

        results_frame = tk.LabelFrame(
            self.root,
            text="INSPECTION REPORT",
            font=("Arial", 11, "bold")
        )

        results_frame.pack(
            fill=tk.X,
            padx=20,
            pady=(3, 12)
        )


        self.results_text = tk.Label(
            results_frame,
            text="No inspection performed.",
            font=("Consolas", 10),
            justify=tk.LEFT,
            anchor="w"
        )

        self.results_text.pack(
            fill=tk.X,
            padx=15,
            pady=8
        )


    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    def display_image(
        self,
        image,
        label
    ):

        display = image.copy()

        max_width = 540
        max_height = 280

        display.thumbnail(
            (max_width, max_height),
            Image.Resampling.LANCZOS
        )

        tk_image = ImageTk.PhotoImage(
            display
        )

        label.configure(
            image=tk_image,
            text=""
        )

        if label == self.original_label:
            self.tk_original = tk_image
        else:
            self.tk_overlay = tk_image


    # ========================================================
    # UPLOAD IMAGE
    # ========================================================

    def upload_image(self):

        path = filedialog.askopenfilename(
            title="Select steel surface image",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"
                ),
                (
                    "All files",
                    "*.*"
                )
            ]
        )

        if not path:
            return


        try:

            image = Image.open(
                path
            ).convert("RGB")

            self.original_image = image


            self.display_image(
                image,
                self.original_label
            )


            self.status_label.config(
                text="Running AI inspection..."
            )

            self.root.update_idletasks()


            prediction, confidence = (
                self.inspector.predict(
                    image
                )
            )

            self.prediction = prediction
            self.confidence = confidence


            overlay = make_overlay(
                image,
                prediction
            )

            self.overlay_image = overlay


            self.display_image(
                overlay,
                self.overlay_label
            )


            self.create_report(
                prediction,
                confidence,
                path
            )


            self.status_label.config(
                text="Inspection complete."
            )


        except Exception as e:

            messagebox.showerror(
                "Inspection Error",
                str(e)
            )

            self.status_label.config(
                text="Inspection failed."
            )


    # ========================================================
    # REPORT
    # ========================================================

    def create_report(
        self,
        prediction,
        confidence,
        image_path
    ):

        # Convert model output back to original image resolution
        prediction_original = resize_prediction_to_original(
            prediction,
            self.original_image.size
        )

        # Confidence must also be resized to the original size
        confidence_tensor = torch.from_numpy(
            confidence
        ).unsqueeze(0).unsqueeze(0).float()

        width, height = self.original_image.size

        confidence_original = F.interpolate(
            confidence_tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()


        total_pixels = prediction_original.size

        lines = []

        lines.append(
            f"Image: {Path(image_path).name}"
        )

        lines.append("")


        detected_classes = []
        possible_classes = []
        class_results = {}


        # ----------------------------------------------------
        # PER-CLASS RESULTS
        # ----------------------------------------------------

        for class_id in range(
            1,
            NUM_CLASSES
        ):

            pixels = int(
                np.sum(
                    prediction_original == class_id
                )
            )

            percentage = (
                100.0
                * pixels
                / total_pixels
            )


            if pixels < MIN_DEFECT_PIXELS:

                state = "IGNORED"

            elif percentage < MIN_DEFECT_AREA_PERCENT:

                state = "IGNORED"

            elif percentage < CONFIRMED_DEFECT_AREA_PERCENT:

                state = "POSSIBLE"
                possible_classes.append(class_id)

            else:

                state = "DETECTED"
                detected_classes.append(class_id)


            class_results[class_id] = {
                "pixels": pixels,
                "percentage": percentage,
                "state": state,
            }


            lines.append(
                f"Class {class_id:<3} "
                f"{CLASS_NAMES[class_id]:<18} "
                f"{state:<14} "
                f"{percentage:>7.3f}%"
            )


        lines.append("")


        # ----------------------------------------------------
        # OVERALL STATUS
        # ----------------------------------------------------

        if detected_classes:

            overall = "DEFECT DETECTED"

        elif possible_classes:

            overall = "POSSIBLE DEFECT"

        else:

            overall = "NO DEFECT DETECTED"


        lines.append(
            f"OVERALL STATUS: {overall}"
        )


        # ----------------------------------------------------
        # DEFECT-PIXEL CONFIDENCE
        # ----------------------------------------------------

        meaningful_defect_mask = np.zeros_like(
            prediction_original,
            dtype=bool
        )

        for class_id in detected_classes + possible_classes:

            meaningful_defect_mask |= (
                prediction_original == class_id
            )


        if np.any(meaningful_defect_mask):

            defect_confidence = float(
                np.mean(
                    confidence_original[
                        meaningful_defect_mask
                    ]
                )
            )

        else:

            defect_confidence = 0.0


        lines.append(
            f"Defect-pixel confidence: "
            f"{defect_confidence * 100:.2f}%"
        )


        # ----------------------------------------------------
        # MODEL CONFIDENCE LEVEL
        # ----------------------------------------------------

        if defect_confidence >= 0.85:

            confidence_level = "HIGH"

        elif defect_confidence >= 0.70:

            confidence_level = "MODERATE"

        elif defect_confidence > 0:

            confidence_level = "LOW"

        else:

            confidence_level = "N/A"


        lines.append(
            f"Model confidence level: {confidence_level}"
        )


        # ----------------------------------------------------
        # PRIMARY DEFECT
        # ----------------------------------------------------

        meaningful_classes = (
            detected_classes
            + possible_classes
        )


        if meaningful_classes:

            primary_class = max(
                meaningful_classes,
                key=lambda c:
                    class_results[c]["percentage"]
            )

            lines.append("")

            lines.append(
                f"PRIMARY DEFECT: "
                f"Class {primary_class} - "
                f"{CLASS_NAMES[primary_class]}"
            )

            lines.append(
                f"Description: "
                f"{CLASS_DESCRIPTIONS[primary_class]}"
            )


        # ----------------------------------------------------
        # DETECTED DEFECT TYPES
        # ----------------------------------------------------

        if meaningful_classes:

            lines.append("")
            lines.append(
                "DEFECT DETAILS:"
            )

            for class_id in meaningful_classes:

                result = class_results[class_id]

                lines.append(
                    f"Class {class_id}: "
                    f"{CLASS_NAMES[class_id]} | "
                    f"{result['state']} | "
                    f"{result['percentage']:.3f}%"
                )

                lines.append(
                    f"  {CLASS_DESCRIPTIONS[class_id]}"
                )


        self.last_report = "\n".join(lines)

        self.results_text.config(
            text=self.last_report
        )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    def save_result(self):

        if self.original_image is None:

            messagebox.showinfo(
                "No result",
                "Upload an image and run "
                "an inspection first."
            )

            return


        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        path = filedialog.asksaveasfilename(
            title="Save inspection overlay",
            initialdir=str(
                OUTPUT_DIR.resolve()
            ),
            defaultextension=".png",
            filetypes=[
                (
                    "PNG image",
                    "*.png"
                )
            ]
        )


        if not path:
            return


        self.overlay_image.save(
            path
        )


        # Also save the text report beside the image.
        report_path = Path(path).with_suffix(
            ".txt"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                self.last_report
            )

            file.write(
                "\n\n"
                "Decision thresholds:\n"
                f"Minimum meaningful area: "
                f"{MIN_DEFECT_AREA_PERCENT:.2f}%\n"
                f"Confirmed defect area: "
                f"{CONFIRMED_DEFECT_AREA_PERCENT:.2f}%\n"
            )


        messagebox.showinfo(
            "Saved",
            "Inspection results saved:\n\n"
            f"Overlay:\n{path}\n\n"
            f"Report:\n{report_path}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        print("=" * 70)
        print(
            "INDUSTRIAL STEEL DEFECT INSPECTION"
        )
        print("=" * 70)
        print()

        print(
            "Loading verified model checkpoint..."
        )

        inspector = SteelInspectionModel()

        print(
            "Model loaded successfully."
        )

        print(
            f"Device: {inspector.device}"
        )

        root = tk.Tk()

        InspectionApp(
            root,
            inspector
        )

        root.mainloop()


    except Exception as e:

        print()
        print("ERROR:")
        print(e)
        print()

        input(
            "Press Enter to exit..."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
