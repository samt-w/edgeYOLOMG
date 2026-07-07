# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ARD100 Raw Data Paths
ARD100_RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR"))
TRAIN_VIDEOS_DIR = ARD100_RAW_DATA_DIR / "train_videos"
TEST_VIDEOS_DIR = ARD100_RAW_DATA_DIR / "test_videos"
ANNOTATIONS_DIR = ARD100_RAW_DATA_DIR / "annotations"

# ARD100 Processed Data Paths (Local SSD)
ARD100_PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR"))
FRAMES_DIR = PROCESSED_DATA_DIR / "images"
MASKS_DIR = PROCESSED_DATA_DIR / "masks"