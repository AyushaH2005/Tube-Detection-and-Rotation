# Microcentrifuge Tube Detection and Orientation Estimation

## Overview

This project detects microcentrifuge tube lids from overhead RGB images and predicts their orientations.

Pipeline:

```text
Image → YOLOv8 Detector → Tube Crop → Angle Model → Final Prediction
```

The system uses:

* YOLOv8 for tube detection
* EfficientNet-B0 for orientation estimation
* sin/cos angle regression for stable angle prediction

---

# Project Structure

```text
project/
│
├── images/                     # Original dataset images
├── annotations.csv             # Ground truth annotations
│
├── prepare_dataset.py          # Converts annotations to YOLO format
├── train_detector.py           # Trains YOLOv8 detector
├── create_angle_dataset.py     # Creates cropped tube images
├── train_angle_model.py        # Trains orientation model
├── inference.py                # Runs prediction on images
├── evaluate.py                 # Computes metrics
├── tube.yaml                   # YOLO dataset config
├── requirements.txt            # Required libraries
│
├── dataset/                    # YOLO train/val dataset
├── crops/                      # Tube crops for angle training
├── models/                     # Saved angle model
├── runs/                       # YOLO training outputs
├── augmented_images/           # Augmented images
└── result.png                  # Final inference visualization
```

---

# File Descriptions

## prepare_dataset.py

Splits dataset into train/validation sets and converts annotations into YOLO label format.

## train_detector.py

Trains YOLOv8 on tube detection.

Output:

```text
runs/detect/tube_detector/weights/best.pt
```

## create_angle_dataset.py

Crops tube regions from images for orientation learning.

## train_angle_model.py

Trains EfficientNet-B0 to predict:

```text
cos(theta), sin(theta)
```

instead of directly predicting angles.

## inference.py

Runs the complete detection + orientation pipeline and saves visualized results.

## evaluate.py

Evaluates:

* Precision
* Recall
* F1 Score
* Mean Angle Error

---

# How to Run

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Prepare dataset

```bash
python prepare_dataset.py
```

## 3. Train detector

```bash
python train_detector.py
```

## 4. Create angle crops

```bash
python create_angle_dataset.py
```

## 5. Train orientation model

```bash
python train_angle_model.py
```

## 6. Run inference

```bash
python inference.py
```

## 7. Evaluate

```bash
python evaluate.py
```

---

# Key Technical Idea

Instead of directly regressing angles, the model predicts:

```text
(cos(theta), sin(theta))
```

This avoids instability caused by angle wraparound near:

```text
0° ≈ 360°
```

The final angle is reconstructed using:

```text
theta = atan2(sin(theta), cos(theta))
```

---

# Augmentation Strategy

To improve robustness on the small dataset, augmentations such as:

* brightness variation
* blur
* noise
* glare simulation
* contrast changes
* perspective distortion

were used.

---

# Final Output

The system produces:

* tube detections
* orientation predictions
* visualization outputs
* evaluation metrics
