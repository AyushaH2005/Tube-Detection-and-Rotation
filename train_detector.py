from ultralytics import YOLO

# =========================================
# LOAD PRETRAINED MODEL
# =========================================

model = YOLO("yolov8n.pt")

# =========================================
# TRAIN
# =========================================

model.train(

    # Dataset YAML
    data="tube.yaml",

    # Training params
    epochs=25,
    imgsz=416,
    batch=4,

    # Save location
    project="runs/detect",

    # Folder name
    name="tube_detector",

    # IMPORTANT:
    # overwrite existing folder
    exist_ok=True,

    # Augmentations
    augment=True,

    degrees=10,
    translate=0.1,
    scale=0.2,
    fliplr=0.5,

    # Early stopping
    patience=10
)

print("\nTraining completed successfully!")
print(
    "Best model saved at:\n"
    "runs/detect/tube_detector/weights/best.pt"
)