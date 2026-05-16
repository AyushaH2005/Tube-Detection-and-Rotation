import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------
# Paths
# --------------------------------

IMAGE_DIR = "augmented_images"
CSV_PATH = "augmented_annotations.csv"

CROPS_DIR = "crops"

# --------------------------------
# Create folders
# --------------------------------

os.makedirs(f"{CROPS_DIR}/train", exist_ok=True)
os.makedirs(f"{CROPS_DIR}/val", exist_ok=True)

# --------------------------------
# Read annotations
# --------------------------------

df = pd.read_csv(CSV_PATH)

# --------------------------------
# Split images
# IMPORTANT:
# split by image, not by rows
# --------------------------------

all_images = df["image"].unique()

train_images, val_images = train_test_split(
    all_images,
    test_size=0.2,
    random_state=42
)

train_images = set(train_images)
val_images = set(val_images)

# --------------------------------
# Store metadata
# --------------------------------

records = []

counter = 0

# --------------------------------
# Process every annotation
# --------------------------------

for _, row in df.iterrows():

    image_name = row["image"]

    image_path = os.path.join(IMAGE_DIR, image_name)

    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not read image: {image_path}")
        continue

    # Bounding box
    x = int(row["bbox_x"])
    y = int(row["bbox_y"])
    w = int(row["bbox_w"])
    h = int(row["bbox_h"])

    # Safety clipping
    x = max(0, x)
    y = max(0, y)

    x2 = min(img.shape[1], x + w)
    y2 = min(img.shape[0], y + h)

    crop = img[y:y2, x:x2]

    # Skip invalid crops
    if crop.size == 0:
        continue

    # Determine split
    if image_name in train_images:
        split = "train"
    else:
        split = "val"

    # Save crop
    crop_filename = f"{counter}.png"

    crop_path = os.path.join(
        CROPS_DIR,
        split,
        crop_filename
    )

    cv2.imwrite(crop_path, crop)

    # Store label
    records.append({
        "file": crop_filename,
        "split": split,
        "angle": row["angle_deg"]
    })

    counter += 1

# --------------------------------
# Save angle labels CSV
# --------------------------------

angle_df = pd.DataFrame(records)

angle_df.to_csv("angle_labels.csv", index=False)

print("Angle dataset created successfully!")
print(f"Total crops created: {len(records)}")