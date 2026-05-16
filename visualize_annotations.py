"""
visualize_annotations.py
========================
Visualization utility for microcentrifuge tube annotations.

Draws bounding boxes, centre points, and orientation arrows on images.
Works on both original and augmented images/CSVs.

Usage
-----
    python visualize_annotations.py                         # default paths
    python visualize_annotations.py --images_dir project/augmented_images \
                                    --csv project/augmented_annotations.csv \
                                    --output_dir project/visualizations \
                                    --max_images 20
"""

import argparse
import math
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

# Colour palette for up to 10 tubes per image (BGR)
_PALETTE = [
    (0, 200, 0),     # green
    (0, 100, 255),   # orange
    (255, 50, 50),   # blue
    (180, 0, 220),   # purple
    (0, 220, 220),   # yellow
    (220, 0, 100),   # pink-blue
    (0, 160, 255),   # gold
    (50, 255, 180),  # lime-cyan
    (255, 150, 0),   # sky blue
    (80, 80, 255),   # salmon
]


def draw_bbox(img: np.ndarray, bbox_x: float, bbox_y: float,
              bbox_w: float, bbox_h: float, color: tuple,
              thickness: int = 2) -> np.ndarray:
    """Draw axis-aligned bounding box."""
    x1, y1 = int(bbox_x), int(bbox_y)
    x2, y2 = int(bbox_x + bbox_w), int(bbox_y + bbox_h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def draw_center(img: np.ndarray, cx: float, cy: float,
                color: tuple, radius: int = 4) -> np.ndarray:
    """Draw filled circle at tube centre."""
    cv2.circle(img, (int(cx), int(cy)), radius, color, -1)
    return img


def draw_orientation_arrow(img: np.ndarray, cx: float, cy: float,
                           angle_deg: float, color: tuple,
                           length: int = 40, thickness: int = 2) -> np.ndarray:
    """
    Draw a double-headed arrow showing tube orientation axis.

    angle_deg is the undirected axis angle [0, 180).
    Both directions are drawn to reflect the undirected nature.
    """
    rad = math.radians(angle_deg)
    dx = math.cos(rad) * length
    dy = math.sin(rad) * length
    # draw arrow in both directions
    pt_fwd = (int(cx + dx), int(cy + dy))
    pt_bwd = (int(cx - dx), int(cy - dy))
    cv2.arrowedLine(img, (int(cx), int(cy)), pt_fwd, color, thickness,
                    tipLength=0.3)
    cv2.arrowedLine(img, (int(cx), int(cy)), pt_bwd, color, thickness,
                    tipLength=0.3)
    return img


def draw_angle_label(img: np.ndarray, cx: float, cy: float,
                     angle_deg: float, color: tuple) -> np.ndarray:
    """Overlay angle text near tube centre."""
    label = f"{angle_deg:.1f}\u00b0"
    pos = (int(cx) + 6, int(cy) - 6)
    # dark outline for readability on any background
    cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                color, 1, cv2.LINE_AA)
    return img


def annotate_image(img: np.ndarray, rows: pd.DataFrame,
                   arrow_length: int = 40) -> np.ndarray:
    """
    Draw all tube annotations from a DataFrame subset onto a copy of img.

    Parameters
    ----------
    img         : BGR image
    rows        : DataFrame rows for this image (one row per tube)
    arrow_length: pixel length of orientation arrows
    """
    vis = img.copy()
    for i, (_, row) in enumerate(rows.iterrows()):
        color = _PALETTE[i % len(_PALETTE)]
        draw_bbox(vis, row["bbox_x"], row["bbox_y"],
                  row["bbox_w"], row["bbox_h"], color)
        draw_center(vis, row["center_x"], row["center_y"], color)
        draw_orientation_arrow(vis, row["center_x"], row["center_y"],
                               row["angle_deg"], color, length=arrow_length)
        draw_angle_label(vis, row["center_x"], row["center_y"],
                         row["angle_deg"], color)
    return vis


# ---------------------------------------------------------------------------
# Batch visualisation
# ---------------------------------------------------------------------------

def visualize_batch(
    images_dir: Path,
    csv_path: Path,
    output_dir: Path,
    max_images: Optional[int] = None,
    arrow_length: int = 40,
    shuffle: bool = True,
    seed: int = 42,
) -> None:
    """
    Read CSV, draw annotations on each image, save to output_dir.

    Parameters
    ----------
    images_dir  : directory containing the images
    csv_path    : annotations CSV
    output_dir  : where visualized images are saved
    max_images  : cap on number of images to process (None = all)
    arrow_length: length of orientation arrows in pixels
    shuffle     : randomly order images (useful for inspection sampling)
    seed        : RNG seed
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    filenames = df["image_filename"].unique().tolist()
    if shuffle:
        random.seed(seed)
        random.shuffle(filenames)
    if max_images is not None:
        filenames = filenames[:max_images]

    ok, skipped = 0, 0
    for fname in filenames:
        img_path = images_dir / fname
        if not img_path.exists():
            print(f"  [WARN] Image not found: {img_path}")
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Could not read: {img_path}")
            skipped += 1
            continue

        rows = df[df["image_filename"] == fname]
        vis = annotate_image(img, rows, arrow_length=arrow_length)

        out_path = output_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)
        ok += 1

    print(f"Visualized {ok} images → {output_dir}  (skipped {skipped})")


def make_contact_sheet(
    images_dir: Path,
    csv_path: Path,
    output_path: Path,
    n_cols: int = 4,
    thumb_size: int = 200,
    max_images: int = 40,
    seed: int = 42,
) -> None:
    """
    Create a single contact-sheet image showing annotated thumbnails.
    Convenient for a quick sanity-check overview.

    Parameters
    ----------
    images_dir  : directory containing the images
    csv_path    : annotations CSV
    output_path : file path for the contact sheet PNG
    n_cols      : number of columns
    thumb_size  : pixel size of each thumbnail (square)
    max_images  : cap on thumbnails included
    seed        : RNG seed
    """
    df = pd.read_csv(csv_path)
    filenames = df["image_filename"].unique().tolist()
    random.seed(seed)
    random.shuffle(filenames)
    filenames = filenames[:max_images]

    thumbs = []
    for fname in filenames:
        img_path = images_dir / fname
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        rows = df[df["image_filename"] == fname]
        vis = annotate_image(img, rows)
        # resize to square thumbnail
        h, w = vis.shape[:2]
        scale = thumb_size / max(h, w)
        vis_small = cv2.resize(vis, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        # pad to thumb_size × thumb_size
        pad = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
        ph = (thumb_size - vis_small.shape[0]) // 2
        pw = (thumb_size - vis_small.shape[1]) // 2
        pad[ph:ph + vis_small.shape[0], pw:pw + vis_small.shape[1]] = vis_small
        # filename label
        cv2.putText(pad, Path(fname).stem[:18], (2, thumb_size - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1, cv2.LINE_AA)
        thumbs.append(pad)

    if not thumbs:
        print("[WARN] No thumbnails generated.")
        return

    n_rows = math.ceil(len(thumbs) / n_cols)
    # pad list to full grid
    while len(thumbs) < n_rows * n_cols:
        thumbs.append(np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8))

    rows_imgs = []
    for r in range(n_rows):
        row_img = np.hstack(thumbs[r * n_cols:(r + 1) * n_cols])
        rows_imgs.append(row_img)
    sheet = np.vstack(rows_imgs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    print(f"Contact sheet ({len(filenames)} images) → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Visualize tube annotations.")
    p.add_argument("--images_dir",   default="project/augmented_images",
                   help="Directory with images to visualize.")
    p.add_argument("--csv",          default="project/augmented_annotations.csv",
                   help="Annotations CSV path.")
    p.add_argument("--output_dir",   default="project/visualizations",
                   help="Where annotated images are saved.")
    p.add_argument("--max_images",   type=int, default=None,
                   help="Maximum number of images to visualize (default: all).")
    p.add_argument("--contact_sheet", action="store_true",
                   help="Also generate a contact-sheet overview image.")
    p.add_argument("--arrow_length", type=int, default=40,
                   help="Length of orientation arrows in pixels.")
    p.add_argument("--no_shuffle",   action="store_true",
                   help="Process images in CSV order, not shuffled.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    images_dir = Path(args.images_dir)
    csv_path   = Path(args.csv)
    output_dir = Path(args.output_dir)

    visualize_batch(
        images_dir=images_dir,
        csv_path=csv_path,
        output_dir=output_dir,
        max_images=args.max_images,
        arrow_length=args.arrow_length,
        shuffle=not args.no_shuffle,
    )

    if args.contact_sheet:
        make_contact_sheet(
            images_dir=images_dir,
            csv_path=csv_path,
            output_path=output_dir / "contact_sheet.png",
        )
