# evaluate.py

import os
import cv2
import math
import torch
import pandas as pd
from PIL import Image

from ultralytics import YOLO
from torchvision import transforms, models

# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# =====================================================
# PATHS
# =====================================================

IMAGE_DIR = "images"

GT_CSV = "annotations.csv"

YOLO_MODEL_PATH = (
    "runs/detect/models/tube_detector/weights/best.pt"
)

ANGLE_MODEL_PATH = "models/angle_model.pth"

# =====================================================
# CHECK FILES
# =====================================================

if not os.path.exists(YOLO_MODEL_PATH):
    raise FileNotFoundError(
        f"\nYOLO model not found:\n{YOLO_MODEL_PATH}"
    )

if not os.path.exists(ANGLE_MODEL_PATH):
    raise FileNotFoundError(
        f"\nAngle model not found:\n{ANGLE_MODEL_PATH}"
    )

if not os.path.exists(GT_CSV):
    raise FileNotFoundError(
        f"\nGround truth CSV not found:\n{GT_CSV}"
    )

# =====================================================
# LOAD YOLO DETECTOR
# =====================================================

print("Loading YOLO detector...")

detector = YOLO(YOLO_MODEL_PATH)

# =====================================================
# LOAD ANGLE MODEL
# =====================================================

print("Loading angle model...")

angle_model = models.resnet18(weights=None)

angle_model.fc = torch.nn.Linear(
    angle_model.fc.in_features,
    2
)

angle_model.load_state_dict(
    torch.load(
        ANGLE_MODEL_PATH,
        map_location=DEVICE
    )
)

angle_model = angle_model.to(DEVICE)

angle_model.eval()

# =====================================================
# IMAGE TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# =====================================================
# PREDICT ANGLE
# =====================================================

def predict_angle(crop):

    image = Image.fromarray(
        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    )

    image = transform(image)

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = angle_model(image)[0]

    cos_val = output[0].item()
    sin_val = output[1].item()

    angle = math.degrees(
        math.atan2(sin_val, cos_val)
    )

    if angle < 0:
        angle += 360

    return angle

# =====================================================
# ANGULAR ERROR
# =====================================================

def angular_error(pred, gt):

    diff = abs(pred - gt) % 360

    return min(diff, 360 - diff)

# =====================================================
# LOAD GROUND TRUTH
# =====================================================

gt_df = pd.read_csv(GT_CSV)

all_images = gt_df["image"].unique()

# =====================================================
# METRICS
# =====================================================

true_positive = 0
false_positive = 0
false_negative = 0

angle_errors = []

DIST_THRESHOLD = 30

# =====================================================
# EVALUATE EACH IMAGE
# =====================================================

for image_name in all_images:

    print(f"\nProcessing: {image_name}")

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not read: {image_path}")
        continue

    # -------------------------------------------------
    # Run detector
    # -------------------------------------------------

    results = detector(img)[0]

    predictions = []

    # -------------------------------------------------
    # Process predictions
    # -------------------------------------------------

    for box in results.boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        angle = predict_angle(crop)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        predictions.append({
            "cx": cx,
            "cy": cy,
            "angle": angle
        })

    # -------------------------------------------------
    # Ground truth rows
    # -------------------------------------------------

    gt_rows = gt_df[
        gt_df["image"] == image_name
    ]

    matched_predictions = set()

    # -------------------------------------------------
    # Match GT with predictions
    # -------------------------------------------------

    for _, gt in gt_rows.iterrows():

        gx = gt["center_x"]
        gy = gt["center_y"]

        best_idx = -1
        best_dist = 1e9

        for idx, pred in enumerate(predictions):

            if idx in matched_predictions:
                continue

            px = pred["cx"]
            py = pred["cy"]

            dist = math.sqrt(
                (gx - px) ** 2 +
                (gy - py) ** 2
            )

            if dist < best_dist:

                best_dist = dist
                best_idx = idx

        # -------------------------------------------------
        # Valid detection
        # -------------------------------------------------

        if best_dist < DIST_THRESHOLD:

            true_positive += 1

            matched_predictions.add(best_idx)

            pred_angle = predictions[best_idx]["angle"]

            gt_angle = gt["angle_deg"]

            err = angular_error(
                pred_angle,
                gt_angle
            )

            angle_errors.append(err)

        else:

            false_negative += 1

    # -------------------------------------------------
    # Remaining predictions = false positives
    # -------------------------------------------------

    false_positive += (
        len(predictions)
        - len(matched_predictions)
    )

# =====================================================
# FINAL METRICS
# =====================================================

precision = true_positive / (
    true_positive + false_positive + 1e-8
)

recall = true_positive / (
    true_positive + false_negative + 1e-8
)

f1 = 2 * precision * recall / (
    precision + recall + 1e-8
)

if len(angle_errors) > 0:
    mean_angle_error = (
        sum(angle_errors) / len(angle_errors)
    )
else:
    mean_angle_error = 0

# =====================================================
# PRINT RESULTS
# =====================================================

print("\n===================================")
print("FINAL EVALUATION RESULTS")
print("===================================")

print(f"True Positives : {true_positive}")
print(f"False Positives: {false_positive}")
print(f"False Negatives: {false_negative}")

print(f"\nPrecision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print(
    f"Mean Angle Error: "
    f"{mean_angle_error:.2f} degrees"
)

print("\nEvaluation completed successfully!")