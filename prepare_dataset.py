import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------

IMAGE_DIR = "augmented_images"
CSV_PATH = "augmented_annotations.csv"

OUT_IMG_DIR = "dataset/images"
OUT_LABEL_DIR = "dataset/labels"

# -----------------------------
# Create folders
# -----------------------------

os.makedirs(f"{OUT_IMG_DIR}/train", exist_ok=True)
os.makedirs(f"{OUT_IMG_DIR}/val", exist_ok=True)

os.makedirs(f"{OUT_LABEL_DIR}/train", exist_ok=True)
os.makedirs(f"{OUT_LABEL_DIR}/val", exist_ok=True)

# -----------------------------
# Read annotations
# -----------------------------

df = pd.read_csv(CSV_PATH)

# -----------------------------
# Split by IMAGE (important)
# -----------------------------

all_images = df["image"].unique()

train_images, val_images = train_test_split(
    all_images,
    test_size=0.2,
    random_state=42
)

split_map = {}

for img in train_images:
    split_map[img] = "train"

for img in val_images:
    split_map[img] = "val"

# -----------------------------
# Helper function
# Convert bbox -> YOLO format
# -----------------------------

def convert_to_yolo(x, y, w, h, img_w, img_h):

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h

    width = w / img_w
    height = h / img_h

    return x_center, y_center, width, height

# -----------------------------
# Process each image
# -----------------------------

for image_name in all_images:

    split = split_map[image_name]

    image_path = os.path.join(IMAGE_DIR, image_name)

    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not read: {image_path}")
        continue

    img_h, img_w = img.shape[:2]

    # Copy image into train/val folder
    out_image_path = os.path.join(
        OUT_IMG_DIR,
        split,
        image_name
    )

    cv2.imwrite(out_image_path, img)

    # Create YOLO label file
    label_lines = []

    rows = df[df["image"] == image_name]

    for _, row in rows.iterrows():

        bbox_x = row["bbox_x"]
        bbox_y = row["bbox_y"]
        bbox_w = row["bbox_w"]
        bbox_h = row["bbox_h"]

        xc, yc, bw, bh = convert_to_yolo(
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
            img_w,
            img_h
        )

        # Class ID = 0 (tube)
        line = f"0 {xc} {yc} {bw} {bh}"

        label_lines.append(line)

    # Save label file
    label_filename = image_name.replace(".png", ".txt")

    label_path = os.path.join(
        OUT_LABEL_DIR,
        split,
        label_filename
    )

    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))

print("Dataset preparation completed successfully!")