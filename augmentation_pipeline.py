"""
augmentation_pipeline.py
========================
Augmentation pipeline for microcentrifuge tube detection & orientation estimation.

Design rationale
----------------
Small dataset (70 images) → goal is maximum diversity without invalidating labels.
All geometric transforms that change bbox / angle are handled with exact label updates.
Photometric transforms are label-safe and applied aggressively.

Augmentation taxonomy used here
--------------------------------
SAFE (no label update needed):
  - Brightness / contrast / gamma
  - Color temperature shift
  - Shadow / illumination gradient overlay
  - Gaussian noise
  - Gaussian / motion blur
  - JPEG compression artifacts
  - Local glare / specular highlight

GEOMETRIC (label update required):
  - Small translation    → bbox center shifts; angle unchanged
  - Small scaling        → bbox center + size scale; angle unchanged
  - Slight perspective   → bbox corners warp; angle changes slightly (tracked)
  - Full rotation        → bbox rotates; angle updates exactly
"""

import os
import cv2
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# All paths are resolved relative to THIS script's location, so the pipeline
# works correctly regardless of which directory you run it from.
# If augmentation_pipeline.py lives inside  project/  (next to annotations.csv),
# the defaults below are correct.  Adjust if your layout differs.

_HERE = Path(__file__).resolve().parent   # directory that contains this script

IMAGES_DIR      = _HERE / "images"
ANNOTATIONS_CSV = _HERE / "annotations.csv"
OUTPUT_IMAGES   = _HERE / "augmented_images"
OUTPUT_CSV      = _HERE / "augmented_annotations.csv"

NUM_AUGMENTATIONS_PER_IMAGE = 5   # at least 5 versions per original
SEED = 42                          # reproducibility

random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Annotation:
    """Single tube annotation within one image."""
    image_filename: str
    center_x: float
    center_y: float
    bbox_x: float          # top-left x
    bbox_y: float          # top-left y
    bbox_w: float
    bbox_h: float
    angle_deg: float       # orientation in degrees [0, 180)

    def bbox_corners(self) -> np.ndarray:
        """Return 4 corners of bbox as (4,2) float32, TL→TR→BR→BL."""
        x, y, w, h = self.bbox_x, self.bbox_y, self.bbox_w, self.bbox_h
        return np.array([
            [x,     y    ],
            [x + w, y    ],
            [x + w, y + h],
            [x,     y + h],
        ], dtype=np.float32)

    def from_corners(corners: np.ndarray, angle_deg: float,
                     image_filename: str) -> "Annotation":
        """Reconstruct Annotation from transformed 4-corners + new angle."""
        xs = corners[:, 0]
        ys = corners[:, 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        return Annotation(
            image_filename=image_filename,
            center_x=cx,
            center_y=cy,
            bbox_x=x_min,
            bbox_y=y_min,
            bbox_w=x_max - x_min,
            bbox_h=y_max - y_min,
            angle_deg=float(np.mod(angle_deg, 360.0)),
        )


def load_annotations(csv_path: Path) -> pd.DataFrame:
    """
    Load and normalise the annotations CSV.

    Your CSV uses column name 'image' (not 'image_filename').
    It also has a 'bbox_rot' column (always 0) which is harmlessly ignored.
    Angles are in [0, 360) (directed); kept that way throughout the pipeline.
    """
    df = pd.read_csv(csv_path)

    # Rename 'image' -> 'image_filename' (or other common variants)
    for candidate in ("image", "filename", "file_name", "img"):
        if candidate in df.columns and "image_filename" not in df.columns:
            df = df.rename(columns={candidate: "image_filename"})
            break

    # Validate required columns
    required = {"image_filename", "center_x", "center_y",
                "bbox_x", "bbox_y", "bbox_w", "bbox_h", "angle_deg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"annotations.csv is missing columns: {missing}  "
            f"Columns found: {list(df.columns)}"
        )

    # Normalise angle to [0, 360) — guards stray values; preserves directed angle
    df["angle_deg"] = df["angle_deg"].astype(float) % 360.0

    return df


def annotations_for_image(df: pd.DataFrame, filename: str) -> List[Annotation]:
    rows = df[df["image_filename"] == filename]
    return [
        Annotation(
            image_filename=row["image_filename"],
            center_x=float(row["center_x"]),
            center_y=float(row["center_y"]),
            bbox_x=float(row["bbox_x"]),
            bbox_y=float(row["bbox_y"]),
            bbox_w=float(row["bbox_w"]),
            bbox_h=float(row["bbox_h"]),
            angle_deg=float(row["angle_deg"]),
        )
        for _, row in rows.iterrows()
    ]

# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def warp_point(pt: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 3×3 homography or 2×3 affine matrix to a 2-D point."""
    if M.shape == (2, 3):
        # affine
        p = np.array([pt[0], pt[1], 1.0])
        return M @ p
    else:
        # perspective
        p = np.array([pt[0], pt[1], 1.0])
        r = M @ p
        return r[:2] / r[2]


def warp_corners(corners: np.ndarray, M: np.ndarray) -> np.ndarray:
    return np.array([warp_point(c, M) for c in corners], dtype=np.float32)


def rotation_angle_update(angle_deg: float, rot_deg: float) -> float:
    """
    Update orientation label after image rotation by rot_deg CCW.
    Angles are directed [0, 360) so we mod by 360.
    """
    new_angle = (angle_deg + rot_deg) % 360.0
    return new_angle


def perspective_angle_update(angle_deg: float,
                              src_center: np.ndarray,
                              M: np.ndarray,
                              delta: float = 5.0) -> float:
    """
    Estimate new orientation after perspective warp by tracking a short
    axis-aligned line segment through the warp matrix.

    Approximation: project two points offset along the tube axis and
    compute the resulting angle.  Works well for small distortions.
    """
    rad = math.radians(angle_deg)
    # direction vector of the tube
    dx = math.cos(rad) * delta
    dy = math.sin(rad) * delta
    p1 = warp_point(src_center, M)
    p2 = warp_point(src_center + np.array([dx, dy]), M)
    diff = p2 - p1
    new_rad = math.atan2(diff[1], diff[0])
    new_deg = math.degrees(new_rad) % 360.0
    return new_deg


def clip_bbox(ann: Annotation, img_h: int, img_w: int) -> Optional[Annotation]:
    """Clip bbox to image bounds; return None if bbox area is too small."""
    x1 = max(0.0, ann.bbox_x)
    y1 = max(0.0, ann.bbox_y)
    x2 = min(float(img_w), ann.bbox_x + ann.bbox_w)
    y2 = min(float(img_h), ann.bbox_y + ann.bbox_h)
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None
    return Annotation(
        image_filename=ann.image_filename,
        center_x=(x1 + x2) / 2.0,
        center_y=(y1 + y2) / 2.0,
        bbox_x=x1, bbox_y=y1,
        bbox_w=x2 - x1, bbox_h=y2 - y1,
        angle_deg=ann.angle_deg,
    )

# ---------------------------------------------------------------------------
# Photometric augmentations  (label-safe)
# ---------------------------------------------------------------------------

def aug_brightness(img: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
    """
    WHY: Microscopy / lab lighting varies. Teaching the model different
    overall exposures improves robustness to under/over-lit conditions.
    RISK: None for orientation or bbox.
    """
    if factor is None:
        factor = random.uniform(0.55, 1.55)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_contrast(img: np.ndarray, alpha: Optional[float] = None,
                 beta: Optional[float] = None) -> np.ndarray:
    """
    WHY: Tube caps have subtle colour/brightness differences from the tube
    body. Varying contrast trains the model to detect structure at multiple
    contrast levels.
    RISK: None for geometry labels.
    """
    if alpha is None:
        alpha = random.uniform(0.6, 1.8)
    if beta is None:
        beta = random.uniform(-30, 30)
    out = img.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_gamma(img: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
    """
    WHY: Camera response curves differ. Gamma correction simulates this.
    """
    if gamma is None:
        gamma = random.uniform(0.5, 2.0)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def aug_color_temperature(img: np.ndarray,
                          shift: Optional[int] = None) -> np.ndarray:
    """
    WHY: Lab cameras have different white-balance settings. Colour-temp
    shifts simulate warm (tungsten) to cool (daylight/LED) environments.
    RISK: None.
    """
    if shift is None:
        shift = random.randint(-40, 40)
    out = img.astype(np.int32)
    # positive shift → warmer (more R, less B)
    out[:, :, 2] = np.clip(out[:, :, 2] + shift, 0, 255)      # R channel (BGR idx 2)
    out[:, :, 0] = np.clip(out[:, :, 0] - shift, 0, 255)      # B channel (BGR idx 0)
    return out.astype(np.uint8)


def aug_shadow(img: np.ndarray) -> np.ndarray:
    """
    WHY: Lab bench shadows from equipment, hands, or overhead fixtures
    create partial occlusion. Random polygon shadows improve robustness.
    RISK: None for labels.
    """
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    # random triangular / quadrilateral shadow
    num_pts = random.randint(3, 5)
    pts = np.array(
        [[random.randint(0, w), random.randint(0, h)] for _ in range(num_pts)],
        dtype=np.int32
    )
    shadow_intensity = random.uniform(0.25, 0.6)
    cv2.fillConvexPoly(mask, pts, shadow_intensity)
    # blur the shadow edge for realism
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    out = img.astype(np.float32)
    out *= mask[:, :, np.newaxis]
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_illumination_gradient(img: np.ndarray) -> np.ndarray:
    """
    WHY: Non-uniform lighting from ring lights, single-source illumination,
    or reflective bench tops creates gradients across the image.
    RISK: None for labels.
    """
    h, w = img.shape[:2]
    direction = random.choice(["h", "v", "diag"])
    strength = random.uniform(0.3, 0.7)

    if direction == "h":
        grad = np.linspace(1.0, 1.0 - strength, w, dtype=np.float32)
        grad = np.tile(grad, (h, 1))
    elif direction == "v":
        grad = np.linspace(1.0, 1.0 - strength, h, dtype=np.float32)
        grad = np.tile(grad.reshape(-1, 1), (1, w))
    else:
        gx = np.linspace(1.0, 1.0 - strength, w, dtype=np.float32)
        gy = np.linspace(1.0, 1.0 - strength, h, dtype=np.float32)
        grad = np.outer(gy, gx)

    if random.random() < 0.5:
        grad = np.flip(grad, axis=random.randint(0, 1))

    out = img.astype(np.float32) * grad[:, :, np.newaxis]
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_gaussian_noise(img: np.ndarray,
                       sigma: Optional[float] = None) -> np.ndarray:
    """
    WHY: Sensor noise varies by camera. Gaussian noise prevents the model
    from relying on pixel-perfect texture patterns.
    RISK: None for labels.
    """
    if sigma is None:
        sigma = random.uniform(3.0, 20.0)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_gaussian_blur(img: np.ndarray,
                      ksize: Optional[int] = None) -> np.ndarray:
    """
    WHY: Slight defocus / camera shake blurs the image. Blurred training
    samples improve detector recall on out-of-focus frames.
    RISK: None for labels.
    """
    if ksize is None:
        ksize = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def aug_motion_blur(img: np.ndarray,
                    kernel_size: Optional[int] = None) -> np.ndarray:
    """
    WHY: Camera or subject motion creates directional blur, especially
    in automated pipelines with moving sample stages.
    RISK: None for labels.
    """
    if kernel_size is None:
        kernel_size = random.choice([5, 7, 9])
    angle = random.uniform(0, 180)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    M_rot = cv2.getRotationMatrix2D(
        (kernel_size / 2, kernel_size / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M_rot, (kernel_size, kernel_size))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def aug_jpeg_compression(img: np.ndarray,
                         quality: Optional[int] = None) -> np.ndarray:
    """
    WHY: Images captured and re-saved as JPEG (common in lab pipelines)
    introduce blocking and ringing artifacts. Training on these prevents
    quality-dependent performance drops.
    RISK: None for labels.
    """
    if quality is None:
        quality = random.randint(40, 85)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def aug_glare(img: np.ndarray) -> np.ndarray:
    """
    WHY: Specular reflections on polished tube surfaces (plastic caps) are
    common under bright or angled lighting. Simulated glare spots train the
    model to handle local saturation.
    RISK: None for labels.
    """
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    num_spots = random.randint(1, 4)
    for _ in range(num_spots):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        radius = random.randint(8, 40)
        intensity = random.uniform(0.4, 1.0)
        # create a bright elliptical spot
        spot = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(spot, (cx, cy), (radius, max(2, radius // 2)),
                    random.uniform(0, 180), 0, 360, intensity, -1)
        spot = cv2.GaussianBlur(spot, (0, 0), radius / 2.0)
        out += spot[:, :, np.newaxis] * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_background_tone(img: np.ndarray) -> np.ndarray:
    """
    WHY: Lab bench surfaces differ in colour (white, grey, blue). A subtle
    colour cast simulates different bench / mat backgrounds visible around
    the tubes.
    RISK: None for labels.
    """
    shift = np.array(
        [random.randint(-25, 25), random.randint(-25, 25), random.randint(-25, 25)],
        dtype=np.int32
    )
    out = img.astype(np.int32) + shift[np.newaxis, np.newaxis, :]
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# Geometric augmentations  (label update required)
# ---------------------------------------------------------------------------

def aug_translate(img: np.ndarray, annotations: List[Annotation],
                  max_px: Optional[int] = None
                  ) -> Tuple[np.ndarray, List[Annotation]]:
    """
    WHY: Tubes rarely appear at the exact same image position. Translation
    teaches position invariance without changing appearance or angle.
    LABEL UPDATE: Add (tx, ty) to all centers and bbox corners.
    RISK: None for orientation angles.
    """
    h, w = img.shape[:2]
    if max_px is None:
        max_px = min(h, w) // 12
    tx = random.randint(-max_px, max_px)
    ty = random.randint(-max_px, max_px)
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    warped = cv2.warpAffine(img, M, (w, h),
                            borderMode=cv2.BORDER_REFLECT_101)
    new_anns = []
    for ann in annotations:
        corners = warp_corners(ann.bbox_corners(), M)
        new_ann = Annotation.from_corners(corners, ann.angle_deg,
                                          ann.image_filename)
        new_ann = clip_bbox(new_ann, h, w)
        if new_ann is not None:
            new_anns.append(new_ann)
    return warped, new_anns


def aug_scale(img: np.ndarray, annotations: List[Annotation],
              scale_range: Optional[Tuple[float, float]] = None
              ) -> Tuple[np.ndarray, List[Annotation]]:
    """
    WHY: Imaging distance / zoom varies. Scale changes teach size invariance.
    LABEL UPDATE: Scale all coordinates from image center. Angle unchanged.
    RISK: None for orientation angles.
    """
    h, w = img.shape[:2]
    if scale_range is None:
        scale_range = (0.82, 1.18)
    scale = random.uniform(*scale_range)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, scale)
    warped = cv2.warpAffine(img, M, (w, h),
                            borderMode=cv2.BORDER_REFLECT_101)
    new_anns = []
    for ann in annotations:
        corners = warp_corners(ann.bbox_corners(), M)
        new_ann = Annotation.from_corners(corners, ann.angle_deg,
                                          ann.image_filename)
        new_ann = clip_bbox(new_ann, h, w)
        if new_ann is not None:
            new_anns.append(new_ann)
    return warped, new_anns


def aug_rotate(img: np.ndarray, annotations: List[Annotation],
               angle_range: Optional[Tuple[float, float]] = None
               ) -> Tuple[np.ndarray, List[Annotation]]:
    """
    WHY: Tubes can be placed at any orientation on the bench; rotation
    augmentation directly trains orientation invariance.
    LABEL UPDATE: Rotate bbox corners by same angle; subtract rotation from
    orientation labels (because tube axis rotates with the image).
    RISK: HIGH – must update angle exactly or predictions will be corrupted.
    NOTE: We rotate the image CCW by rot_deg, so the tube axis seen in the
    image also rotates CCW by rot_deg → new label = old_label + rot_deg.
    """
    h, w = img.shape[:2]
    if angle_range is None:
        angle_range = (-30.0, 30.0)
    rot_deg = random.uniform(*angle_range)
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -rot_deg, 1.0)
    # compute new canvas size so nothing is cut off
    cos_a = abs(math.cos(math.radians(rot_deg)))
    sin_a = abs(math.sin(math.radians(rot_deg)))
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    warped = cv2.warpAffine(img, M, (new_w, new_h),
                            borderMode=cv2.BORDER_REFLECT_101)
    # resize back to original dims
    warped = cv2.resize(warped, (w, h), interpolation=cv2.INTER_LINEAR)
    # compute combined M that also rescales back
    scale_x = w / new_w
    scale_y = h / new_h
    M_scale = np.array([[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32)
    # compose: first M_rot (2×3 → 3×3), then M_scale (2×3 → 3×3)
    M_rot_3 = np.vstack([M, [0, 0, 1]])
    M_scale_3 = np.vstack([M_scale, [0, 0, 1]])
    M_full_3 = M_scale_3 @ M_rot_3
    M_full = M_full_3[:2, :]
    new_anns = []
    for ann in annotations:
        corners = warp_corners(ann.bbox_corners(), M_full)
        new_angle = rotation_angle_update(ann.angle_deg, rot_deg)
        new_ann = Annotation.from_corners(corners, new_angle,
                                          ann.image_filename)
        new_ann = clip_bbox(new_ann, h, w)
        if new_ann is not None:
            new_anns.append(new_ann)
    return warped, new_anns


def aug_perspective(img: np.ndarray, annotations: List[Annotation],
                    distortion: Optional[float] = None
                    ) -> Tuple[np.ndarray, List[Annotation]]:
    """
    WHY: Camera is never perfectly orthogonal to the bench. Slight
    perspective teaches the model to handle non-orthographic viewpoints
    and improves 3-D pose awareness.
    LABEL UPDATE: Each bbox corner warped through homography; angle
    estimated by tracking the tube axis direction through the warp.
    RISK: MEDIUM – angle update is approximate; keep distortion small.
    """
    h, w = img.shape[:2]
    if distortion is None:
        distortion = random.uniform(0.01, 0.06)
    # perturb corners of a slightly inset region
    margin = int(min(h, w) * 0.05)
    src_pts = np.float32([
        [margin,     margin    ],
        [w - margin, margin    ],
        [w - margin, h - margin],
        [margin,     h - margin],
    ])
    def rp() -> float:
        return random.uniform(-distortion, distortion)
    dst_pts = np.float32([
        [src_pts[0][0] + rp() * w, src_pts[0][1] + rp() * h],
        [src_pts[1][0] + rp() * w, src_pts[1][1] + rp() * h],
        [src_pts[2][0] + rp() * w, src_pts[2][1] + rp() * h],
        [src_pts[3][0] + rp() * w, src_pts[3][1] + rp() * h],
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h),
                                 borderMode=cv2.BORDER_REFLECT_101)
    new_anns = []
    for ann in annotations:
        corners = warp_corners(ann.bbox_corners(), M)
        src_center = np.array([ann.center_x, ann.center_y], dtype=np.float32)
        new_angle = perspective_angle_update(ann.angle_deg, src_center, M)
        new_ann = Annotation.from_corners(corners, new_angle,
                                          ann.image_filename)
        new_ann = clip_bbox(new_ann, h, w)
        if new_ann is not None:
            new_anns.append(new_ann)
    return warped, new_anns

# ---------------------------------------------------------------------------
# Augmentation combinator
# ---------------------------------------------------------------------------

# Pool of photometric transforms (all label-safe)
_PHOTO_TRANSFORMS = [
    aug_brightness,
    aug_contrast,
    aug_gamma,
    aug_color_temperature,
    aug_shadow,
    aug_illumination_gradient,
    aug_gaussian_noise,
    aug_gaussian_blur,
    aug_motion_blur,
    aug_jpeg_compression,
    aug_glare,
    aug_background_tone,
]

# Geometric transforms with their weight (higher = more likely)
_GEO_TRANSFORMS = [
    (aug_translate,   0.7),
    (aug_scale,       0.5),
    (aug_rotate,      0.6),
    (aug_perspective, 0.4),
]


def _weighted_choice(transforms_weights):
    names, weights = zip(*[(t, w) for t, w in transforms_weights])
    probs = np.array(weights, dtype=float)
    probs /= probs.sum()
    return list(np.random.choice(names, size=len(names), replace=False, p=probs))


def build_augmentation(
    img: np.ndarray,
    annotations: List[Annotation],
    aug_index: int,
    filename_stem: str,
) -> Tuple[np.ndarray, List[Annotation]]:
    """
    Build one augmented sample by randomly combining transforms.

    Strategy for small-data diversity:
      - Always apply 2-4 photometric transforms (safe)
      - Apply 1-2 geometric transforms with probability-based selection
      - Vary the combination per aug_index to reduce repetition
    """
    rng_state = random.getstate()
    np_state = np.random.get_state()

    # deterministic per (image, index) so reruns are consistent
    seed_val = hash(filename_stem + str(aug_index)) % (2 ** 31)
    random.seed(seed_val)
    np.random.seed(seed_val)

    result_img = img.copy()
    result_anns = [a for a in annotations]
    out_filename = f"{filename_stem}_aug_{aug_index}.png"
    for ann in result_anns:
        ann.image_filename = out_filename

    # --- Photometric (always applied) ---
    num_photo = random.randint(3, 6)
    photo_pool = random.sample(_PHOTO_TRANSFORMS, min(num_photo, len(_PHOTO_TRANSFORMS)))
    for transform_fn in photo_pool:
        result_img = transform_fn(result_img)

    # --- Geometric (applied probabilistically) ---
    geo_ordered = _weighted_choice(_GEO_TRANSFORMS)
    num_geo = random.randint(1, 2)
    geo_selected = geo_ordered[:num_geo]
    for transform_fn in geo_selected:
        try:
            result_img, result_anns = transform_fn(result_img, result_anns)
            # update filenames after each warp
            for ann in result_anns:
                ann.image_filename = out_filename
        except Exception as exc:
            # If a geometric transform fails for any edge reason, skip it
            print(f"  [WARN] Geometric transform {transform_fn.__name__} failed: {exc}")

    # Restore global RNG state so main loop is not affected
    random.setstate(rng_state)
    np.random.set_state(np_state)

    return result_img, result_anns

# ---------------------------------------------------------------------------
# Main pipeline entry-point
# ---------------------------------------------------------------------------

def run_pipeline(
    images_dir: Path = IMAGES_DIR,
    annotations_csv: Path = ANNOTATIONS_CSV,
    output_images: Path = OUTPUT_IMAGES,
    output_csv: Path = OUTPUT_CSV,
    num_augmentations: int = NUM_AUGMENTATIONS_PER_IMAGE,
    copy_originals: bool = True,
) -> pd.DataFrame:
    """
    Execute the full augmentation pipeline.

    Parameters
    ----------
    images_dir        : directory of source images
    annotations_csv   : CSV with original annotations
    output_images     : where augmented images are saved
    output_csv        : path for the combined annotation CSV
    num_augmentations : augmented versions per original image
    copy_originals    : if True, also copy originals into output with their annotations

    Returns
    -------
    DataFrame with all (original + augmented) annotations.
    """
    output_images.mkdir(parents=True, exist_ok=True)

    df = load_annotations(annotations_csv)
    unique_files = df["image_filename"].unique().tolist()
    print(f"Found {len(unique_files)} unique images in annotations.")

    all_records: List[Dict] = []

    # Optionally carry originals through
    if copy_originals:
        for fname in unique_files:
            src = images_dir / fname
            if not src.exists():
                print(f"  [WARN] Original not found, skipping copy: {src}")
                continue
            dst = output_images / fname
            img = cv2.imread(str(src))
            if img is None:
                continue
            cv2.imwrite(str(dst), img)
        for _, row in df.iterrows():
            rec = row.to_dict()
            # Rename internal key back to original CSV column name
            if "image_filename" in rec and "image" not in rec:
                rec["image"] = rec.pop("image_filename")
            all_records.append(rec)

    # Augmentation loop
    for fname in unique_files:
        img_path = images_dir / fname
        if not img_path.exists():
            print(f"  [WARN] Image not found: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Could not read image: {img_path}")
            continue

        stem = Path(fname).stem
        annotations = annotations_for_image(df, fname)

        print(f"Augmenting {fname} ({len(annotations)} tubes) …")

        for aug_idx in range(1, num_augmentations + 1):
            aug_img, aug_anns = build_augmentation(
                img, annotations, aug_idx, stem
            )
            if not aug_anns:
                print(f"  [WARN] aug_{aug_idx}: all annotations clipped out, skipping.")
                continue

            out_fname = f"{stem}_aug_{aug_idx}.png"
            out_path = output_images / out_fname
            cv2.imwrite(str(out_path), aug_img)

            for ann in aug_anns:
                all_records.append({
                    "image":      out_fname,          # match original CSV column name
                    "center_x":   round(ann.center_x, 3),
                    "center_y":   round(ann.center_y, 3),
                    "bbox_x":     round(ann.bbox_x, 3),
                    "bbox_y":     round(ann.bbox_y, 3),
                    "bbox_w":     round(ann.bbox_w, 3),
                    "bbox_h":     round(ann.bbox_h, 3),
                    "bbox_rot":   0,                  # always 0 (no in-plane rotation baked into bbox)
                    "angle_deg":  round(ann.angle_deg, 4),
                })

    result_df = pd.DataFrame(all_records)
    result_df.to_csv(output_csv, index=False)
    print(f"\nDone. {len(result_df)} total annotation rows → {output_csv}")
    print(f"Augmented images → {output_images}")
    return result_df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()