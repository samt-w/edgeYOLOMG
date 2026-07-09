import cv2
import os
import sys
from pathlib import Path

# connect to config by adding the parent directory of the current working directory 
# to the list of paths where Python searches for modules
sys.path.append('..')
from config import TRAIN_VIDEOS_DIR, TEST_VIDEOS_DIR, IMAGES_DIR

def process_video_frames(video_path: Path):
    """
    This function extracts individual RGB frames from a single video
    """
    video_name = video_path.stem
    print(f"Processing frames from: {video_name}")

    # Construct the save path using pathlib
    save_dir = IMAGES_DIR / video_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # create an OpenCV VideoCapture object
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    while cap.isOpened():
        frame_count += 1
        # when the loop reaches the end of the video, return_status is False and the loop breaks
        return_status, currentFrame = cap.read()
        if not return_status:
            break

        # Write each frame to the corresponding folder
        filename = f"{video_name}_{str(frame_count).zfill(4)}.jpg"
        cv2.imwrite(str(save_dir / filename), currentFrame)
    cap.release()

def process_video_directory_frames(video_dir: Path):
    # this function scans the given directory for mp4 files and calls the motion mask function on them
    if not video_dir.exists():
        print(f"Directory not found: {video_dir}")
        return
    
    # extract list of all mp4 files in directory
    video_files = list(video_dir.glob("*.mp4"))

    if not video_files:
        print(f"No .mp4 files found in {video_dir}")
        return
    
    for video_path in video_files:
        process_video_frames(video_path)

def process_single_video_frames(video_path: Path):
    """
    THIS FUNCTION IS JUST FOR TESTING PURPOSES
    It only processes a single video and calls the frame extraction function on it
    """
    if not video_path.exists():
        print(f"File not found: {video_path}")
        return
        
    print(f"Processing single video: {video_path.name}")
    
    process_video_frames(video_path)

if __name__ == "__main__":
    # To test code on a single video first
    test_video_path = TRAIN_VIDEOS_DIR / "phantom09.mp4"
    print("--- Testing Single Video ---")
    process_single_video_frames(test_video_path)
    
    # print("--- Processing Training Videos ---")
    # process_video_directory_frames(TRAIN_VIDEOS_DIR)
    # print("\n--- Processing Test Videos ---")
    # process_video_directory_frames(TEST_VIDEOS_DIR)