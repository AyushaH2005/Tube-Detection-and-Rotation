# inference.py

import os
import cv2
import math
import torch
from PIL import Image

from ultralytics import YOLO
from torchvision import transforms, models

# =========================================
# DEVICE
# =========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# =========================================
# PATHS
# =========================================

# YOLO trained model path
YOLO_MODEL_PATH = "runs/detect/models/tube_detector/weights/best.pt"

# Angle model path
ANGLE_MODEL_PATH = "models/angle_model.pth"

# Input image path
# CHANGE THIS TO ANY IMAGE FROM YOUR DATASET

IMAGE_PATH = "images/2659ffa5-color.png"

# Output image
OUTPUT_PATH = "result.png"

# =========================================
# CHECK FILES
# =========================================

if not os.path.exists(YOLO_MODEL_PATH):
    raise FileNotFoundError(
        f"\nYOLO model not found:\n{YOLO_MODEL_PATH}"
    )

if not os.path.exists(ANGLE_MODEL_PATH):
    raise FileNotFoundError(
        f"\nAngle model not found:\n{ANGLE_MODEL_PATH}"
    )

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"\nImage not found:\n{IMAGE_PATH}"
    )

# =========================================
# LOAD YOLO DETECTOR
# =========================================

print("Loading YOLO detector...")

detector = YOLO(YOLO_MODEL_PATH)

# =========================================
# LOAD ANGLE MODEL
# =========================================

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

# =========================================
# IMAGE TRANSFORM
# =========================================

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# =========================================
# ANGLE PREDICTION FUNCTION
# =========================================

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

# =========================================
# LOAD IMAGE
# =========================================

print("Loading image...")

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError(
        f"\nCould not read image:\n{IMAGE_PATH}"
    )

# =========================================
# RUN DETECTOR
# =========================================

print("Running detector...")

results = detector(img)[0]

print(f"Detected {len(results.boxes)} tubes")

# =========================================
# PROCESS DETECTIONS
# =========================================

for idx, box in enumerate(results.boxes):

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0].tolist()
    )

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    # -------------------------------------
    # Predict angle
    # -------------------------------------

    angle = predict_angle(crop)

    # -------------------------------------
    # Center point
    # -------------------------------------

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # -------------------------------------
    # Draw bounding box
    # -------------------------------------

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # -------------------------------------
    # Draw orientation arrow
    # -------------------------------------

    length = 40

    end_x = int(
        cx + length * math.cos(math.radians(angle))
    )

    end_y = int(
        cy - length * math.sin(math.radians(angle))
    )

    cv2.arrowedLine(
        img,
        (cx, cy),
        (end_x, end_y),
        (0, 0, 255),
        3
    )

    # -------------------------------------
    # Draw text
    # -------------------------------------

    cv2.putText(
        img,
        f"{angle:.1f} deg",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    print(
        f"Tube {idx+1}: "
        f"Angle = {angle:.2f} deg"
    )

# =========================================
# SAVE OUTPUT
# =========================================

cv2.imwrite(OUTPUT_PATH, img)

print("\nInference completed successfully!")
print(f"Saved result image to: {OUTPUT_PATH}")