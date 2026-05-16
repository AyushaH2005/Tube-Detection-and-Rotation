"""
sanity_check_annotations.py
============================
Sanity-check script for the augmented annotations CSV.

Checks performed
----------------
1.  Required columns present
2.  No NaN values in any column
3.  angle_deg in [0, 180)
4.  bbox_w > 0 and bbox_h > 0
5.  center_x consistent with bbox (within tolerance)
6.  center_y consistent with bbox (within tolerance)
7.  Bbox is within image bounds (image file must exist)
8.  Image files referenced in CSV exist on disk
9.  No image files in output directory are missing from CSV
10. Duplicate (image_filename, center_x, center_y) rows flagged
11. Angle distribution statistics (mean, std per original image stem)
12. Bbox area statistics to detect collapsed boxes

Usage
-----
    python sanity_check_annotations.py                             # defaults
    python sanity_check_annotations.py \
        --images_dir project/augmented_images \
        --csv        project/augmented_annotations.csv \
        --tol        3.0        # pixel tolerance for centre check
        --verbose
"""

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_columns(df: pd.DataFrame) -> List[str]:
    required = {"image_filename", "center_x", "center_y",
                "bbox_x", "bbox_y", "bbox_w", "bbox_h", "angle_deg"}
    missing = required - set(df.columns)
    if missing:
        return [f"FAIL  Missing columns: {missing}"]
    return ["PASS  All required columns present"]


def check_no_nan(df: pd.DataFrame) -> List[str]:
    issues = []
    for col in ["center_x", "center_y", "bbox_x", "bbox_y",
                "bbox_w", "bbox_h", "angle_deg"]:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n > 0:
            issues.append(f"FAIL  Column '{col}' has {n} NaN value(s)")
    if not issues:
        issues.append("PASS  No NaN values in numeric columns")
    return issues


def check_angle_range(df: pd.DataFrame) -> List[str]:
    bad = df[(df["angle_deg"] < 0) | (df["angle_deg"] >= 180)]
    if len(bad):
        return [f"FAIL  {len(bad)} row(s) with angle_deg outside [0, 180): "
                f"indices {list(bad.index[:10])}"]
    return ["PASS  All angle_deg values in [0, 180)"]


def check_bbox_positive(df: pd.DataFrame) -> List[str]:
    issues = []
    bad_w = df[df["bbox_w"] <= 0]
    bad_h = df[df["bbox_h"] <= 0]
    if len(bad_w):
        issues.append(f"FAIL  {len(bad_w)} row(s) with bbox_w <= 0")
    if len(bad_h):
        issues.append(f"FAIL  {len(bad_h)} row(s) with bbox_h <= 0")
    if not issues:
        issues.append("PASS  All bbox dimensions positive")
    return issues


def check_center_consistency(df: pd.DataFrame, tol: float = 3.0) -> List[str]:
    """
    centre should equal (bbox_x + bbox_w/2, bbox_y + bbox_h/2).
    After perspective warps the computed bbox is the axis-aligned bounding box
    of the warped corners, so small discrepancies are expected. Use tol pixels.
    """
    expected_cx = df["bbox_x"] + df["bbox_w"] / 2.0
    expected_cy = df["bbox_y"] + df["bbox_h"] / 2.0
    err_x = (df["center_x"] - expected_cx).abs()
    err_y = (df["center_y"] - expected_cy).abs()
    bad = df[(err_x > tol) | (err_y > tol)]
    if len(bad):
        return [f"WARN  {len(bad)} row(s) where centre deviates "
                f"> {tol}px from bbox midpoint (check perspective warp rows)"]
    return [f"PASS  All centres within {tol}px of bbox midpoint"]


def check_bbox_bounds(df: pd.DataFrame, images_dir: Path,
                      verbose: bool = False) -> List[str]:
    """Check that bboxes lie within image dimensions."""
    issues = []
    ok = 0
    missing_img = 0
    oob = 0
    for fname, group in df.groupby("image_filename"):
        img_path = images_dir / fname
        if not img_path.exists():
            missing_img += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            missing_img += 1
            continue
        h, w = img.shape[:2]
        for _, row in group.iterrows():
            x2 = row["bbox_x"] + row["bbox_w"]
            y2 = row["bbox_y"] + row["bbox_h"]
            if (row["bbox_x"] < -1 or row["bbox_y"] < -1
                    or x2 > w + 1 or y2 > h + 1):
                oob += 1
                if verbose:
                    issues.append(
                        f"  OOB: {fname}  bbox=({row['bbox_x']:.1f},{row['bbox_y']:.1f},"
                        f"{row['bbox_w']:.1f},{row['bbox_h']:.1f})  img={w}x{h}")
            else:
                ok += 1
    if oob:
        issues.insert(0, f"FAIL  {oob} bbox(es) exceed image dimensions")
    else:
        issues.append(f"PASS  All {ok} bboxes within image bounds "
                      f"(skipped {missing_img} missing images)")
    return issues


def check_image_files_exist(df: pd.DataFrame, images_dir: Path) -> List[str]:
    """Every image_filename in CSV should have a file on disk."""
    missing = [f for f in df["image_filename"].unique()
               if not (images_dir / f).exists()]
    if missing:
        sample = missing[:5]
        return [f"FAIL  {len(missing)} image(s) in CSV not on disk. "
                f"Sample: {sample}"]
    return ["PASS  All images in CSV exist on disk"]


def check_orphan_images(df: pd.DataFrame, images_dir: Path) -> List[str]:
    """Every image in images_dir should appear in CSV."""
    on_disk = {p.name for p in images_dir.glob("*.png")}
    in_csv  = set(df["image_filename"].unique())
    orphans = on_disk - in_csv
    if orphans:
        sample = list(orphans)[:5]
        return [f"WARN  {len(orphans)} image(s) on disk not in CSV. "
                f"Sample: {sample}"]
    return ["PASS  No orphan images on disk (all appear in CSV)"]


def check_duplicates(df: pd.DataFrame) -> List[str]:
    dup = df.duplicated(subset=["image_filename", "center_x", "center_y"])
    n = dup.sum()
    if n:
        return [f"WARN  {n} duplicate (image_filename, center_x, center_y) row(s)"]
    return ["PASS  No duplicate annotation rows"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_statistics(df: pd.DataFrame) -> None:
    print("\n─── Annotation Statistics ─────────────────────────────────────")
    total_imgs = df["image_filename"].nunique()
    total_rows = len(df)
    print(f"  Total images    : {total_imgs}")
    print(f"  Total tube anns : {total_rows}")
    print(f"  Avg tubes/image : {total_rows / max(total_imgs, 1):.2f}")

    # Angle distribution
    angles = df["angle_deg"].dropna().values
    print(f"\n  Angle (deg) — min={angles.min():.1f}  max={angles.max():.1f}  "
          f"mean={angles.mean():.1f}  std={angles.std():.1f}")

    # bbox area distribution
    areas = (df["bbox_w"] * df["bbox_h"]).dropna().values
    print(f"  BBox area (px²) — min={areas.min():.0f}  max={areas.max():.0f}  "
          f"mean={areas.mean():.0f}  std={areas.std():.0f}")

    # Count per original stem
    df["_stem"] = df["image_filename"].apply(
        lambda x: x.split("_aug_")[0] if "_aug_" in x else x.rsplit(".", 1)[0]
    )
    counts = df.groupby("_stem")["image_filename"].nunique()
    print(f"\n  Augmented versions per original — "
          f"min={counts.min()}  max={counts.max()}  mean={counts.mean():.1f}")

    # Detect potential collapsed bboxes
    tiny = (areas < 16).sum()
    if tiny:
        print(f"\n  WARN: {tiny} bbox(es) with area < 16 px² (possibly collapsed)")

    print("────────────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_checks(images_dir: Path, csv_path: Path,
               tol: float = 3.0, verbose: bool = False) -> bool:
    """
    Run all checks.  Returns True if no FAIL detected, False otherwise.
    """
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return False
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        return False

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}\n")

    all_messages = []
    all_messages += check_columns(df)
    all_messages += check_no_nan(df)
    all_messages += check_angle_range(df)
    all_messages += check_bbox_positive(df)
    all_messages += check_center_consistency(df, tol=tol)
    all_messages += check_image_files_exist(df, images_dir)
    all_messages += check_orphan_images(df, images_dir)
    all_messages += check_duplicates(df)
    all_messages += check_bbox_bounds(df, images_dir, verbose=verbose)

    has_fail = False
    for msg in all_messages:
        prefix = msg.split()[0]
        if prefix == "FAIL":
            print(f"  ❌ {msg}")
            has_fail = True
        elif prefix == "WARN":
            print(f"  ⚠️  {msg}")
        else:
            print(f"  ✅ {msg}")

    print_statistics(df)

    if has_fail:
        print("\n[RESULT] ❌ Annotation check FAILED – see FAIL lines above.")
    else:
        print("\n[RESULT] ✅ All hard checks passed.")

    return not has_fail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Sanity-check tube annotations.")
    p.add_argument("--images_dir", default="project/augmented_images",
                   help="Directory containing the images.")
    p.add_argument("--csv",        default="project/augmented_annotations.csv",
                   help="Annotations CSV to check.")
    p.add_argument("--tol",        type=float, default=3.0,
                   help="Pixel tolerance for centre-vs-bbox consistency check.")
    p.add_argument("--verbose",    action="store_true",
                   help="Print per-row details for out-of-bounds bboxes.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ok = run_checks(
        images_dir=Path(args.images_dir),
        csv_path=Path(args.csv),
        tol=args.tol,
        verbose=args.verbose,
    )
    sys.exit(0 if ok else 1)
