import cv2
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5
TARGET_HEIGHT = 256
TARGET_WIDTH = 1024

CHECKPOINT_PATH = Path("runs/segmentation_sampling/best_model.pth")

# Decision thresholds used by the live inspector.
MIN_DEFECT_AREA_PERCENT = 0.05
CONFIRMED_DEFECT_AREA_PERCENT = 0.50
MIN_DEFECT_PIXELS = 20

# Camera settings.
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Run inference at most this many times per second.
# The camera display itself remains continuous.
MAX_INFERENCE_FPS = 10.0

# Overlay opacity.
OVERLAY_ALPHA = 0.45

# The training images are extremely wide (1600x256, aspect ratio 6.25).
# A normal webcam is 16:9, so NEVER squash the complete webcam frame into
# 1024x256. We crop a wide inspection strip first, preserving the geometry
# the model saw during training.
TRAINING_ASPECT = TARGET_WIDTH / TARGET_HEIGHT

# Automatic "is this actually a steel surface?" gate.
# This is deliberately conservative because the segmentation model was
# trained on steel images and can otherwise hallucinate defects on faces,
# walls, furniture, etc.
STEEL_GATE_ENABLED = True
STEEL_GATE_MIN_LOW_SAT_FRACTION = 0.62
STEEL_GATE_MAX_MEAN_SATURATION = 68.0
STEEL_GATE_MIN_EDGE_DENSITY = 0.008

# Require several consecutive steel frames before enabling AI overlay.
STEEL_CONFIRM_FRAMES = 4
STEEL_LOST_FRAMES = 3

# Defect predictions are also temporally filtered so a single noisy frame
# cannot immediately become a confirmed defect.
DEFECT_CONFIRM_FRAMES = 3
DEFECT_LOST_FRAMES = 3


# ============================================================
# DEFECT CLASSES
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
# SAME COLORS AS IMAGE GUI
# ============================================================

# Class 0 = Background
# Class 1 = Yellow   -> Pitted Surface
# Class 2 = Blue     -> Crazing
# Class 3 = Purple   -> Scratches
# Class 4 = Red      -> Patches

PALETTE_RGB = np.array([
    [0, 0, 0],
    [255, 255, 0],
    [0, 100, 255],
    [160, 32, 240],
    [255, 0, 0],
], dtype=np.uint8)

PALETTE_BGR = PALETTE_RGB[:, ::-1].copy()


# ============================================================
# CAMERA GEOMETRY / STEEL-SURFACE GATE
# ============================================================

def crop_to_training_aspect(frame):
    """Center-crop the camera frame to the 1600:256 training aspect ratio."""
    h, w = frame.shape[:2]
    current_aspect = w / float(h)

    if current_aspect > TRAINING_ASPECT:
        crop_w = int(round(h * TRAINING_ASPECT))
        x0 = max(0, (w - crop_w) // 2)
        crop = frame[:, x0:x0 + crop_w]
        return crop, (x0, 0, crop_w, h)

    crop_h = int(round(w / TRAINING_ASPECT))
    y0 = max(0, (h - crop_h) // 2)
    crop = frame[y0:y0 + crop_h, :]
    return crop, (0, y0, w, crop_h)


def steel_surface_score(frame):
    """
    Lightweight rejection gate.

    The segmentation network itself is NOT a steel-vs-background classifier.
    This gate rejects strongly coloured/non-metallic scenes before allowing
    its defect predictions onto the display.

    Returns:
        is_steel, score, diagnostics
    """
    crop, _ = crop_to_training_aspect(frame)

    # Work at a smaller resolution for speed.
    small = cv2.resize(crop, (640, 102), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    saturation = hsv[:, :, 1].astype(np.float32)
    mean_sat = float(np.mean(saturation))
    low_sat_fraction = float(np.mean(saturation < STEEL_GATE_MAX_MEAN_SATURATION))

    # Edge density captures the fine texture typical of the steel-strip
    # imagery while avoiding a costly feature extractor.
    edges = cv2.Canny(gray, 50, 120)
    edge_density = float(np.mean(edges > 0))

    sat_score = np.clip(
        (STEEL_GATE_MIN_LOW_SAT_FRACTION - low_sat_fraction + 0.25) / 0.25,
        0.0, 1.0
    )
    color_score = np.clip(
        (STEEL_GATE_MAX_MEAN_SATURATION + 20.0 - mean_sat) / 20.0,
        0.0, 1.0
    )
    edge_score = np.clip(
        edge_density / 0.04,
        0.0, 1.0
    )

    score = float(
        0.55 * sat_score +
        0.30 * color_score +
        0.15 * edge_score
    )

    is_steel = (
        low_sat_fraction >= STEEL_GATE_MIN_LOW_SAT_FRACTION
        and mean_sat <= STEEL_GATE_MAX_MEAN_SATURATION
        and edge_density >= STEEL_GATE_MIN_EDGE_DENSITY
    )

    diagnostics = {
        "mean_sat": mean_sat,
        "low_sat_fraction": low_sat_fraction,
        "edge_density": edge_density,
        "score": score,
    }

    return is_steel, score, diagnostics


class TemporalGate:

    def __init__(self):
        self.steel_good = 0
        self.steel_bad = 0
        self.steel_confirmed = False

    def update(self, steel_now):
        if steel_now:
            self.steel_good += 1
            self.steel_bad = 0
            if self.steel_good >= STEEL_CONFIRM_FRAMES:
                self.steel_confirmed = True
        else:
            self.steel_bad += 1
            self.steel_good = 0
            if self.steel_bad >= STEEL_LOST_FRAMES:
                self.steel_confirmed = False

        return self.steel_confirmed


class DefectTemporalFilter:

    def __init__(self):
        self.confirm_counts = {c: 0 for c in range(1, NUM_CLASSES)}
        self.lost_counts = {c: 0 for c in range(1, NUM_CLASSES)}
        self.confirmed = {c: False for c in range(1, NUM_CLASSES)}

    def update(self, analysis):
        detected = set(analysis["detected"])

        for c in range(1, NUM_CLASSES):
            if c in detected:
                self.confirm_counts[c] += 1
                self.lost_counts[c] = 0
                if self.confirm_counts[c] >= DEFECT_CONFIRM_FRAMES:
                    self.confirmed[c] = True
            else:
                self.lost_counts[c] += 1
                self.confirm_counts[c] = 0
                if self.lost_counts[c] >= DEFECT_LOST_FRAMES:
                    self.confirmed[c] = False

        return dict(self.confirmed)


# ============================================================
# MODEL
# ============================================================

class LiveSteelInspector:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("=" * 70)
        print("LIVE INDUSTRIAL STEEL DEFECT INSPECTION")
        print("=" * 70)
        print()
        print(f"Device: {self.device}")

        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        print()
        print("Loading trained model...")

        self.model = create_model(
            num_classes=NUM_CLASSES,
            pretrained=False
        )

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                "Trained checkpoint was not found:\n"
                f"{CHECKPOINT_PATH.resolve()}\n\n"
                "Run this program from the project root."
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

        print("Model loaded successfully.")
        print()


    def preprocess(self, frame_bgr):

        # Preserve the training geometry. The model was trained on very wide
        # 1600x256 steel-strip images, so crop the webcam frame to the same
        # aspect ratio instead of vertically squashing it.
        crop, _ = crop_to_training_aspect(frame_bgr)

        frame_rgb = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )

        tensor = torch.from_numpy(
            frame_rgb
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


    @torch.inference_mode()
    def predict(self, frame_bgr):

        tensor = self.preprocess(frame_bgr)

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
# RESIZE PREDICTION TO CAMERA FRAME
# ============================================================

def resize_prediction(
    prediction,
    frame_width,
    frame_height
):

    # Prediction is for the center-cropped training-aspect image.
    crop_width = frame_width
    crop_height = int(round(frame_width / TRAINING_ASPECT))

    crop_height = min(crop_height, frame_height)
    crop_width = min(
        frame_width,
        int(round(crop_height * TRAINING_ASPECT))
    )

    tensor = torch.from_numpy(
        prediction
    ).unsqueeze(0).unsqueeze(0).float()

    tensor = F.interpolate(
        tensor,
        size=(crop_height, crop_width),
        mode="nearest"
    )

    crop_prediction = (
        tensor
        .squeeze()
        .long()
        .numpy()
    )

    full = np.zeros(
        (frame_height, frame_width),
        dtype=np.int64
    )

    x0 = (frame_width - crop_width) // 2
    y0 = (frame_height - crop_height) // 2

    full[
        y0:y0 + crop_height,
        x0:x0 + crop_width
    ] = crop_prediction

    return full


def resize_confidence(
    confidence,
    frame_width,
    frame_height
):

    crop_height = min(
        frame_height,
        int(round(frame_width / TRAINING_ASPECT))
    )
    crop_width = min(
        frame_width,
        int(round(crop_height * TRAINING_ASPECT))
    )

    tensor = torch.from_numpy(
        confidence
    ).unsqueeze(0).unsqueeze(0).float()

    tensor = F.interpolate(
        tensor,
        size=(crop_height, crop_width),
        mode="bilinear",
        align_corners=False
    )

    crop_confidence = tensor.squeeze().numpy()

    full = np.zeros(
        (frame_height, frame_width),
        dtype=np.float32
    )

    x0 = (frame_width - crop_width) // 2
    y0 = (frame_height - crop_height) // 2

    full[
        y0:y0 + crop_height,
        x0:x0 + crop_width
    ] = crop_confidence

    return full


# ============================================================
# LIVE OVERLAY
# ============================================================

def create_overlay(
    frame,
    prediction_original
):

    overlay = frame.copy()

    for class_id in range(1, NUM_CLASSES):

        mask = prediction_original == class_id

        if not np.any(mask):
            continue

        color = PALETTE_BGR[class_id].tolist()

        color_layer = np.zeros_like(
            frame,
            dtype=np.uint8
        )

        color_layer[:, :] = color

        overlay[mask] = (
            (1.0 - OVERLAY_ALPHA)
            * overlay[mask]
            +
            OVERLAY_ALPHA
            * color_layer[mask]
        ).astype(np.uint8)

    return overlay


# ============================================================
# ANALYZE FRAME
# ============================================================

def analyze_prediction(
    prediction_original,
    confidence_original
):

    total_pixels = prediction_original.size

    results = {}

    detected = []
    possible = []

    for class_id in range(1, NUM_CLASSES):

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
            possible.append(class_id)

        else:
            state = "DETECTED"
            detected.append(class_id)

        results[class_id] = {
            "pixels": pixels,
            "percentage": percentage,
            "state": state,
        }

    meaningful = detected + possible

    if detected:
        overall = "DEFECT DETECTED"
    elif possible:
        overall = "POSSIBLE DEFECT"
    else:
        overall = "NO DEFECT DETECTED"

    if meaningful:

        defect_mask = np.zeros_like(
            prediction_original,
            dtype=bool
        )

        for class_id in meaningful:
            defect_mask |= (
                prediction_original == class_id
            )

        defect_confidence = float(
            np.mean(
                confidence_original[
                    defect_mask
                ]
            )
        )

        primary_class = max(
            meaningful,
            key=lambda c:
                results[c]["percentage"]
        )

    else:

        defect_confidence = 0.0
        primary_class = None

    if defect_confidence >= 0.85:
        confidence_level = "HIGH"
    elif defect_confidence >= 0.70:
        confidence_level = "MODERATE"
    elif defect_confidence > 0:
        confidence_level = "LOW"
    else:
        confidence_level = "N/A"

    return {
        "results": results,
        "detected": detected,
        "possible": possible,
        "overall": overall,
        "defect_confidence": defect_confidence,
        "confidence_level": confidence_level,
        "primary_class": primary_class,
    }


# ============================================================
# DRAW INFORMATION PANEL
# ============================================================

def draw_panel(
    frame,
    analysis,
    fps,
    steel_confirmed,
    steel_score,
    diagnostics
):

    height, width = frame.shape[:2]

    panel_width = min(
        430,
        max(350, width // 3)
    )

    panel = np.zeros(
        (height, panel_width, 3),
        dtype=np.uint8
    )

    panel[:] = (25, 25, 25)

    # Header
    cv2.putText(
        panel,
        "LIVE STEEL INSPECTION",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        panel,
        f"AI inference: {fps:.1f} FPS",
        (18, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (190, 190, 190),
        1,
        cv2.LINE_AA
    )

    # Steel-surface status
    if STEEL_GATE_ENABLED and not steel_confirmed:
        cv2.putText(
            panel,
            "SURFACE: NOT STEEL / UNSUITABLE",
            (18, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 180, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            panel,
            f"Steel gate score: {steel_score:.2f}",
            (18, 113),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (190, 190, 190),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            panel,
            "Point camera at the steel surface",
            (18, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (190, 190, 190),
            1,
            cv2.LINE_AA
        )
        y = 175
        status = "NO DEFECT DETECTED"
    else:
        cv2.putText(
            panel,
            "SURFACE: STEEL DETECTED",
            (18, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (120, 255, 120),
            2,
            cv2.LINE_AA
        )
        y = 120

    # Overall status
    status = analysis["overall"]

    if status == "DEFECT DETECTED":
        status_text = "STATUS: DEFECT DETECTED"
    elif status == "POSSIBLE DEFECT":
        status_text = "STATUS: POSSIBLE DEFECT"
    else:
        status_text = "STATUS: NO DEFECT"

    if steel_confirmed or not STEEL_GATE_ENABLED:
        cv2.putText(
            panel,
            status_text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        y += 35
    else:
        cv2.putText(
            panel,
            "AI DEFECT ANALYSIS: BLOCKED",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 180, 255),
            1,
            cv2.LINE_AA
        )
        y += 35

    # Primary defect
    primary = analysis["primary_class"] if (steel_confirmed or not STEEL_GATE_ENABLED) else None

    if primary is not None:

        name = CLASS_NAMES[primary]

        cv2.putText(
            panel,
            f"Detected: {name}",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            PALETTE_BGR[primary].tolist(),
            2,
            cv2.LINE_AA
        )

        y += 27

        cv2.putText(
            panel,
            CLASS_DESCRIPTIONS[primary],
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (205, 205, 205),
            1,
            cv2.LINE_AA
        )

        y += 38

    else:

        cv2.putText(
            panel,
            "Detected: None",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            2,
            cv2.LINE_AA
        )

        y += 40

    # Confidence
    confidence = (
        analysis["defect_confidence"]
        * 100.0
    )

    cv2.putText(
        panel,
        f"Defect confidence: {confidence:.1f}%",
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
        cv2.LINE_AA
    )

    y += 25

    cv2.putText(
        panel,
        f"Confidence level: {analysis['confidence_level']}",
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    y += 38

    # Class-by-class results
    cv2.putText(
        panel,
        "DEFECT CLASSES",
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    y += 28

    for class_id in range(1, NUM_CLASSES):

        result = analysis["results"][class_id]

        state = result["state"]
        percent = result["percentage"]

        if state == "DETECTED":
            label = "DETECTED"
        elif state == "POSSIBLE":
            label = "POSSIBLE"
        else:
            label = "NOT DETECTED"

        # Color swatch
        cv2.rectangle(
            panel,
            (18, y - 13),
            (35, y + 4),
            PALETTE_BGR[class_id].tolist(),
            -1
        )

        text = (
            f"C{class_id} {CLASS_NAMES[class_id]}: "
            f"{label}"
        )

        cv2.putText(
            panel,
            text,
            (45, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (235, 235, 235),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            panel,
            f"{percent:.2f}%",
            (panel_width - 75, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (190, 190, 190),
            1,
            cv2.LINE_AA
        )

        y += 30

    # Legend
    y += 12

    cv2.putText(
        panel,
        "COLOR LEGEND",
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    y += 27

    for class_id in range(1, NUM_CLASSES):

        cv2.rectangle(
            panel,
            (18, y - 13),
            (35, y + 4),
            PALETTE_BGR[class_id].tolist(),
            -1
        )

        cv2.putText(
            panel,
            f"Class {class_id}: {CLASS_NAMES[class_id]}",
            (45, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        y += 25

    # Gate diagnostics
    if STEEL_GATE_ENABLED:
        cv2.putText(
            panel,
            "Gate: AUTO",
            (18, height - 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (150, 150, 150),
            1,
            cv2.LINE_AA
        )

    # Controls
    cv2.putText(
        panel,
        "Q / ESC  =  Exit",
        (18, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    combined = np.hstack(
        (frame, panel)
    )

    return combined


# ============================================================
# MAIN LIVE LOOP
# ============================================================

def main():

    inspector = LiveSteelInspector()

    print(
        f"Opening camera {CAMERA_INDEX}..."
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    if not camera.isOpened():

        raise RuntimeError(
            "Could not open the camera.\n\n"
            "Try changing CAMERA_INDEX = 0 to 1 "
            "near the top of live_camera.py."
        )

    print("Camera opened successfully.")
    print()
    print("Live inference started.")
    print("Camera geometry: center-cropped to 6.25:1 training aspect ratio.")
    print("Steel gate: ENABLED.")
    print("Press Q or ESC to stop.")
    print()

    last_inference_time = 0.0

    steel_gate = TemporalGate()
    defect_filter = DefectTemporalFilter()

    latest_prediction = None
    latest_confidence = None
    latest_analysis = {
        "results": {
            c: {
                "pixels": 0,
                "percentage": 0.0,
                "state": "IGNORED"
            }
            for c in range(1, NUM_CLASSES)
        },
        "detected": [],
        "possible": [],
        "overall": "NO DEFECT DETECTED",
        "defect_confidence": 0.0,
        "confidence_level": "N/A",
        "primary_class": None,
    }

    latest_steel_confirmed = False
    latest_steel_score = 0.0
    latest_gate_diag = {
        "mean_sat": 0.0,
        "low_sat_fraction": 0.0,
        "edge_density": 0.0,
        "score": 0.0,
    }

    inference_count = 0
    inference_times = []

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print(
                    "Failed to read frame from camera."
                )
                break

            current_time = time.perf_counter()

            # ------------------------------------------------
            # Continuous display, controlled inference rate
            # ------------------------------------------------

            if (
                current_time - last_inference_time
                >= 1.0 / MAX_INFERENCE_FPS
            ):

                inference_start = time.perf_counter()

                steel_now, steel_score, gate_diag = (
                    steel_surface_score(frame)
                )

                steel_confirmed = steel_gate.update(
                    steel_now
                )

                if steel_confirmed or not STEEL_GATE_ENABLED:

                    prediction, confidence = (
                        inspector.predict(frame)
                    )

                    frame_height, frame_width = frame.shape[:2]

                    prediction_original = (
                        resize_prediction(
                            prediction,
                            frame_width,
                            frame_height
                        )
                    )

                    confidence_original = (
                        resize_confidence(
                            confidence,
                            frame_width,
                            frame_height
                        )
                    )

                    latest_prediction = (
                        prediction_original
                    )

                    latest_confidence = (
                        confidence_original
                    )

                    latest_analysis = (
                        analyze_prediction(
                            prediction_original,
                            confidence_original
                        )
                    )

                    confirmed_classes = defect_filter.update(
                        latest_analysis
                    )

                    # Only show defects that persist for multiple AI frames.
                    stable_detected = [
                        c for c in range(1, NUM_CLASSES)
                        if confirmed_classes[c]
                    ]

                    stable_possible = [
                        c for c in latest_analysis["possible"]
                        if not confirmed_classes[c]
                    ]

                    latest_analysis["detected"] = stable_detected
                    latest_analysis["possible"] = stable_possible
                    latest_analysis["overall"] = (
                        "DEFECT DETECTED"
                        if stable_detected
                        else (
                            "POSSIBLE DEFECT"
                            if stable_possible
                            else "NO DEFECT DETECTED"
                        )
                    )

                    meaningful = stable_detected + stable_possible

                    if meaningful:
                        latest_analysis["primary_class"] = max(
                            meaningful,
                            key=lambda c: latest_analysis["results"][c]["percentage"]
                        )
                    else:
                        latest_analysis["primary_class"] = None

                else:
                    # Critical safety behavior: when the scene does not look
                    # like the steel domain, do NOT run or display segmentation.
                    latest_prediction = None
                    latest_confidence = None
                    latest_analysis = {
                        "results": {
                            c: {
                                "pixels": 0,
                                "percentage": 0.0,
                                "state": "IGNORED"
                            }
                            for c in range(1, NUM_CLASSES)
                        },
                        "detected": [],
                        "possible": [],
                        "overall": "NO DEFECT DETECTED",
                        "defect_confidence": 0.0,
                        "confidence_level": "N/A",
                        "primary_class": None,
                    }

                latest_steel_confirmed = steel_confirmed
                latest_steel_score = steel_score
                latest_gate_diag = gate_diag

                inference_end = time.perf_counter()

                elapsed = (
                    inference_end
                    - inference_start
                )

                inference_times.append(
                    elapsed
                )

                if len(inference_times) > 20:
                    inference_times.pop(0)

                inference_count += 1
                last_inference_time = current_time

            # ------------------------------------------------
            # Draw current segmentation
            # ------------------------------------------------

            if (
                latest_prediction is not None
                and (latest_steel_confirmed or not STEEL_GATE_ENABLED)
            ):

                display_frame = create_overlay(
                    frame,
                    latest_prediction
                )

            else:

                display_frame = frame.copy()

            # Draw the inspection strip so the operator knows which part of
            # the webcam image is actually being analyzed.
            frame_h, frame_w = display_frame.shape[:2]
            crop_h = min(
                frame_h,
                int(round(frame_w / TRAINING_ASPECT))
            )
            y0 = (frame_h - crop_h) // 2
            cv2.rectangle(
                display_frame,
                (0, y0),
                (frame_w - 1, y0 + crop_h - 1),
                (120, 120, 120),
                1
            )


            # ------------------------------------------------
            # Calculate actual recent inference FPS
            # ------------------------------------------------

            if inference_times:

                average_time = np.mean(
                    inference_times
                )

                inference_fps = (
                    1.0 / average_time
                    if average_time > 0
                    else 0.0
                )

            else:

                inference_fps = 0.0


            display = draw_panel(
                display_frame,
                latest_analysis,
                inference_fps,
                latest_steel_confirmed,
                latest_steel_score,
                latest_gate_diag
            )


            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            cv2.imshow(
                "Industrial Steel - LIVE AI Inspection",
                display
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break


    finally:

        camera.release()
        cv2.destroyAllWindows()

        print()
        print("=" * 70)
        print("LIVE INSPECTION STOPPED")
        print("=" * 70)
        print(
            f"AI inference cycles: {inference_count}"
        )


if __name__ == "__main__":
    main()
