"""
Verification script for checking the amended YOLO labels following 
letterbox() preprocessing to resize the original images to 1280 and 640 pixels

The script does a numeric check but also outputs bounding boxes overlaid
on the sample images for a visual check
"""

import random
from pathlib import Path
import sys

import cv2
import numpy as np

# ----------------------------- CONFIG -------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

ORIG_LABELS_DIR = BASE_DIR / "data_processed_ARD100/labels/train"
ORIG_IMAGES_DIR = BASE_DIR / "data_processed_ARD100/images/train"

RESOLUTIONS = {
    640: {
        "images": BASE_DIR / "data_processed_ARD100/images_640/train",
        "masks": BASE_DIR / "data_processed_ARD100/masks_640/train",
        "labels": BASE_DIR / "data_processed_ARD100/labels_640/train",
    },
    1280: {
        "images": BASE_DIR / "data_processed_ARD100/images_1280/train",
        "masks": BASE_DIR / "data_processed_ARD100/masks_1280/train",
        "labels": BASE_DIR / "data_processed_ARD100/labels_1280/train",
    },
}

OVERLAY_OUT_DIR = BASE_DIR / "data_prep_scripts/verify_overlays"
N_SAMPLES = 25          
PIXEL_TOL = 2.0         
RANDOM_SEED = 0
# ---------------------------------------------------------------------- --

def validate_paths():
    if not ORIG_LABELS_DIR.exists():
        sys.exit(f"[ERROR] Cannot find original labels directory:\n{ORIG_LABELS_DIR}")
    if not ORIG_IMAGES_DIR.exists():
        sys.exit(f"[ERROR] Cannot find original images directory:\n{ORIG_IMAGES_DIR}")
    for res, paths in RESOLUTIONS.items():
        for key, path in paths.items():
            if not path.exists():
                sys.exit(f"[ERROR] Cannot find {res} {key} directory:\n{path}")

def read_yolo_labels(label_path):
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = line.split()
        out.append((int(cls), float(x), float(y), float(w), float(h)))
    return out

def yolo_to_pixel_xyxy(box, img_w, img_h):
    cls, x, y, w, h = box
    x, w = x * img_w, w * img_w
    y, h = y * img_h, h * img_h
    return np.array([x - w / 2, x + w / 2, y - h / 2, y + h / 2])

def expected_letterbox_transform(orig_w, orig_h, new_size, scaleup=True):
    r = min(new_size / orig_h, new_size / orig_w)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad_w = round(orig_w * r)
    new_unpad_h = round(orig_h * r)
    dw = (new_size - new_unpad_w) / 2
    dh = (new_size - new_unpad_h) / 2
    return r, dw, dh

def transform_box(box_xyxy, r, dw, dh):
    xmin, xmax, ymin, ymax = box_xyxy
    return np.array([
        xmin * r + dw,
        xmax * r + dw,
        ymin * r + dh,
        ymax * r + dh,
    ])

def numeric_check():
    random.seed(RANDOM_SEED)
    n_checked_total = 0
    n_failed_total = 0

    for size, paths in RESOLUTIONS.items():
        target_label_files = sorted(paths["labels"].glob("*.txt"))
        
        if not target_label_files:
            print(f"[{size}] No .txt files found in target directory, skipping.")
            continue
            
        # sample based on images available in the resized directory
        sample = random.sample(target_label_files, min(N_SAMPLES, len(target_label_files)))

        n_checked = 0
        n_failed = 0

        for target_label_path in sample:
            base = target_label_path.stem
            orig_label_path = ORIG_LABELS_DIR / f"{base}.txt"
            img_path = ORIG_IMAGES_DIR / f"{base}.jpg"
            
            if not img_path.exists() or not orig_label_path.exists():
                print(f"[{size}] [skip] missing orig image or label for {base}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[{size}] [skip] unreadable {img_path}")
                continue
            h0, w0 = img.shape[:2]

            orig_boxes = read_yolo_labels(orig_label_path)
            target_boxes = read_yolo_labels(target_label_path)

            if len(target_boxes) != len(orig_boxes):
                print(f"[{size}] [FAIL] {base}: box count mismatch (orig={len(orig_boxes)}, target={len(target_boxes)})")
                n_failed += 1
                continue

            r, dw, dh = expected_letterbox_transform(w0, h0, size)

            for ob, tb in zip(orig_boxes, target_boxes):
                orig_px = yolo_to_pixel_xyxy(ob, w0, h0)
                expected_px = transform_box(orig_px, r, dw, dh)
                actual_px = yolo_to_pixel_xyxy(tb, size, size)

                n_checked += 1
                if not np.allclose(expected_px, actual_px, atol=PIXEL_TOL):
                    n_failed += 1
                    print(f"[{size}] [FAIL] {base}: expected {expected_px.round(1)}, got {actual_px.round(1)}")

                x, y, w, h = tb[1:]
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and w > 0 and h > 0):
                    print(f"[{size}] [FAIL] {base}: out-of-range/degenerate box {tb}")
                    n_failed += 1

        print(f"Numeric check [{size}]: {n_checked} boxes checked, {n_failed} failures.")
        n_checked_total += n_checked
        n_failed_total += n_failed
        
    print(f"\nTotal Numeric check: {n_checked_total} boxes checked, {n_failed_total} failures.")

def visual_check():
    random.seed(RANDOM_SEED + 1)
    
    for size, paths in RESOLUTIONS.items():
        out_dir = OVERLAY_OUT_DIR / str(size)
        out_dir.mkdir(parents=True, exist_ok=True)

        label_files = sorted(paths["labels"].glob("*.txt"))
        if not label_files:
            continue
            
        sample = random.sample(label_files, min(N_SAMPLES, len(label_files)))

        for label_path in sample:
            base = label_path.stem
            boxes = read_yolo_labels(label_path)
            
            # Draw on RGB Image
            img_path = paths["images"] / f"{base}.jpg"
            img = cv2.imread(str(img_path))
            if img is not None:
                for b in boxes:
                    xmin, xmax, ymin, ymax = yolo_to_pixel_xyxy(b, size, size)
                    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.imwrite(str(out_dir / f"{base}_rgb.jpg"), img)

            # Draw on Motion Mask
            mask_path = paths["masks"] / f"{base}.jpg" 
            mask = cv2.imread(str(mask_path))
            if mask is not None:
                for b in boxes:
                    xmin, xmax, ymin, ymax = yolo_to_pixel_xyxy(b, size, size)
                    cv2.rectangle(mask, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                cv2.imwrite(str(out_dir / f"{base}_mask.jpg"), mask)

    print(f"Overlay images written to {OVERLAY_OUT_DIR.resolve()}")

if __name__ == "__main__":
    validate_paths()
    print("Paths validated successfully.\n")
    print("Running numeric cross-check against an independent letterbox re-derivation...")
    numeric_check()
    print("\nWriting visual overlays for spot-checking...")
    visual_check()