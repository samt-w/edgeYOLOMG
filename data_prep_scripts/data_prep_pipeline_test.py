## THIS SCRIPT RUNS A TEST FOR THE FOLLOWING DATA PREPARATION SCRIPTS
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
    TRAIN_DESTS, VAL_DESTS, TEST_DESTS
)

# Import functions from pipeline scripts
from extract_frames import process_video_frames
from generate_motion_masks import process_video_masks
from generate_dataset import (
    train_videos, val_videos, test_videos,
    ensure_dirs, process_dataset_split
)
from generate_txts import generate_all_txts

def main():
    pipeline_start_time = time.time()

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
    
    num_videos = len(video_paths)

    # extract frames for the 6 selected videos
    print("--- Extracting frames in parallel ---")
    frames_start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers = RECOMMENDED_CORES) as executor:
        executor.map(process_video_frames, video_paths)
    frames_duration = time.time() - frames_start_time
    
    # generate motion masks for the 6 selected videos
    print("--- Generating masks in parallel ---")
    masks_start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers = RECOMMENDED_CORES) as executor:
        executor.map(process_video_masks, video_paths)
    masks_duration = time.time() - masks_start_time

    # generate the dataset structure for the selected videos
    print("--- Generating Dataset Structure ---")
    dataset_start_time = time.time()
    ensure_dirs()
    
    print("Processing Training subset...")
    process_dataset_split(test_train_vids, TRAIN_DESTS)
    
    print("Processing Validation subset...")
    process_dataset_split(test_val_vids, VAL_DESTS)
    
    print("Processing Test subset...")
    process_dataset_split(test_test_vids, TEST_DESTS)
    dataset_duration = time.time() - dataset_start_time

    # generate the .txt files for the YOLO training .yaml files 
    print("--- Generating YOLO .txt files ---")
    txt_start_time = time.time()
    generate_all_txts()
    txt_duration = time.time() - txt_start_time

    pipeline_duration = time.time() - pipeline_start_time

    # timing report
    print("\n" + "=" * 20)
    print("PREPROCESSING PIPELINE TIMING REPORT")
    print("=" * 20)
    print(f"Total videos processed:     {num_videos}")
    print(f"Time to extract frames:     {frames_duration:.2f} seconds")
    print(f"Time to generate masks:     {masks_duration:.2f} seconds")
    print(f"Time to generate dataset:   {dataset_duration:.2f} seconds")
    print(f"Time to generate txt files: {txt_duration:.2f} seconds")
    print("-" * 20)
    print(f"TOTAL PIPELINE DURATION:    {pipeline_duration:.2f} seconds")
    print("=" * 20)

if __name__ == "__main__":
    main()