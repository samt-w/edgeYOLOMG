## THIS SCRIPT RUNS THE FULL DATA PREPARATION PIPELINE
## extract_frames.py
## generate_motion_masks.py
## generate_dataset.py

import sys
import time
import concurrent.futures
# connect to config by adding the parent directory of the current working directory 
# to the list of paths where Python searches for modules
sys.path.append('..')
# Import paths from config
from config import (
    RECOMMENDED_CORES,
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
    pipeline_start_time = time.time()

    video_paths = []
    
    # add training and validation video paths
    for vid in train_videos + val_videos:
        video_paths.append(TRAIN_VIDEOS_DIR / f"{vid}.mp4")
        
    # add test video paths
    for vid in test_videos:
        video_paths.append(TEST_VIDEOS_DIR / f"{vid}.mp4")
    
    num_videos = len(video_paths)

    # extract frames for the videos
    print("--- Extracting frames in parallel ---")
    frames_start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers = RECOMMENDED_CORES) as executor:
        executor.map(process_video_frames, video_paths)
    frames_duration = time.time() - frames_start_time
    
    # generate motion masks for the videos
    print("--- Generating masks in parallel ---")
    masks_start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers = RECOMMENDED_CORES) as executor:
        executor.map(process_video_masks, video_paths)
    masks_duration = time.time() - masks_start_time

    # 4. Generate the dataset structure for the selected videos
    print("--- Generating Dataset Structure ---")
    dataset_start_time = time.time()
    ensure_dirs()
    
    print("Processing Training subset...")
    process_dataset(train_videos, IMAGES_TRAIN_DIR, MASKS_TRAIN_DIR, LABELS_TRAIN_DIR)
    
    print("Processing Validation subset...")
    process_dataset(val_videos, IMAGES_VAL_DIR, MASKS_VAL_DIR, LABELS_VAL_DIR)
    
    print("Processing Test subset...")
    process_dataset(test_videos, IMAGES_TEST_DIR, MASKS_TEST_DIR, LABELS_TEST_DIR)
    dataset_duration = time.time() - dataset_start_time

    pipeline_duration = time.time() - pipeline_start_time

    # timing report
    print("\n" + "=" * 20)
    print("PREPROCESSING PIPELINE TIMING REPORT")
    print("=" * 20)
    print(f"Total videos processed:     {num_videos}")
    print(f"Time to extract frames:     {frames_duration:.2f} seconds")
    print(f"Time to generate masks:     {masks_duration:.2f} seconds")
    print(f"Time to generate dataset:   {dataset_duration:.2f} seconds")
    print("-" * 20)
    print(f"TOTAL PIPELINE DURATION:    {pipeline_duration:.2f} seconds")
    print("=" * 20)

if __name__ == "__main__":
    main()