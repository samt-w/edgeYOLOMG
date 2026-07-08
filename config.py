# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Raw Data Paths
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR"))
TRAIN_VIDEOS_DIR = RAW_DATA_DIR / "train_videos"
TEST_VIDEOS_DIR = RAW_DATA_DIR / "test_videos"
ANNOTATIONS_DIR = RAW_DATA_DIR / "annotations"

# Processed Data Paths
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR"))

# Modality directories
IMAGES_DIR = PROCESSED_DATA_DIR / "images"
MASKS_DIR = PROCESSED_DATA_DIR / "masks"
LABELS_DIR = PROCESSED_DATA_DIR / "labels"

# Train splits
IMAGES_TRAIN_DIR = IMAGES_DIR / "train"
MASKS_TRAIN_DIR = MASKS_DIR / "train"
LABELS_TRAIN_DIR = LABELS_DIR / "train"

# Val splits
IMAGES_VAL_DIR = IMAGES_DIR / "val"
MASKS_VAL_DIR = MASKS_DIR / "val"
LABELS_VAL_DIR = LABELS_DIR / "val"

# Test splits
IMAGES_TEST_DIR = IMAGES_DIR / "test"
MASKS_TEST_DIR = MASKS_DIR / "test"
LABELS_TEST_DIR = LABELS_DIR / "test"