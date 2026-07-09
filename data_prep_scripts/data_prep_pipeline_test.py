## THIS SCRIPT RUNS A TEST FOR THE FOLLOWING DATA PREPARATION SCRIPTS
## extract_frames.py
## generate_motion_masks.py
## generate_dataset.py

from pathlib import Path
import sys
import concurrent.futures
# connect to config by adding the parent directory of the current working directory 
# to the list of paths where Python searches for modules
sys.path.append('..')
# Import paths from config
from config import (
    TRAIN_VIDEOS_DIR, TEST_VIDEOS_DIR,
    IMAGES_TRAIN_DIR, MASKS_TRAIN_DIR, LABELS_TRAIN_DIR,
    IMAGES_VAL_DIR, MASKS_VAL_DIR, LABELS_VAL_DIR,
    IMAGES_TEST_DIR, MASKS_TEST_DIR, LABELS_TEST_DIR
)

# Import functions from pipeline scripts
from extract_frames import process_video_frames
from generate_motion_masks import process_video_masks
from generate_dataset import (
    train_videos, val_videos, test_videos,
    ensure_dirs, process_dataset
)

def main():
    # select 2 videos from each split
    test_train_vids = train_videos[:2]
    test_val_vids = val_videos[:2]
    test_test_vids = test_videos[:2]

    video_paths = []
    
    # add training and validation video paths
    for vid in test_train_vids + test_val_vids:
        video_paths.append(TRAIN_VIDEOS_DIR / f"{vid}.mp4")
        
    # add test video paths
    for vid in test_test_vids:
        video_paths.append(TEST_VIDEOS_DIR / f"{vid}.mp4")

    # extract frames for the 6 selected videos
    print("--- Extracting frames in parallel ---")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_video_frames, video_paths)
    
    # generate motion masks for the 6 selected videos
    print("--- Generating masks in parallel ---")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_video_frames, video_paths)

    # 4. Generate the dataset structure for the selected videos
    print("\n--- Generating Dataset Structure ---")
    ensure_dirs()
    
    print("Processing Training subset...")
    process_dataset(test_train_vids, IMAGES_TRAIN_DIR, MASKS_TRAIN_DIR, LABELS_TRAIN_DIR)
    
    print("Processing Validation subset...")
    process_dataset(test_val_vids, IMAGES_VAL_DIR, MASKS_VAL_DIR, LABELS_VAL_DIR)
    
    print("Processing Test subset...")
    process_dataset(test_test_vids, IMAGES_TEST_DIR, MASKS_TEST_DIR, LABELS_TEST_DIR)
    
    print("\nTest pipeline complete!")

if __name__ == "__main__":
    main()