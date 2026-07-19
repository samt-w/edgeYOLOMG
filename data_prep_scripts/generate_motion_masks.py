import cv2
import sys
from collections import deque
from pathlib import Path
from motion_mask import FD5_mask
import concurrent.futures

# connect to config by adding the parent directory of the current working directory 
# to the list of paths where Python searches for modules
sys.path.append('..')
from config import TRAIN_VIDEOS_DIR, TEST_VIDEOS_DIR

def process_video_masks(video_path: Path):
    """
    This function processes frames and generate masks for a single video
    """
    video_name = video_path.stem
    print(f"Processing motion masks for: {video_name}")

    # create an OpenCV VideoCapture object
    cap = cv2.VideoCapture(str(video_path))

    # use Python's deque function to manage the frame buffer
    frame_buffer = deque(maxlen = 5)
    frame_count = 0

    # while the capture is initialised 
    while cap.isOpened():
        # when the loop reaches the end of the video, return_status is False and the loop breaks
        return_status, currentFrame = cap.read()
        # add a trigger if the frame is corrupted
        if not return_status or currentFrame is None:
            break

        # add the next frame to the buffer    
        frame_count += 1
        frame_buffer.append(currentFrame)

        # avoid computing mask until have reached 5 frames
        if len(frame_buffer) < 5:
            continue

        # target frame is at index [2]; the mask is comparing frames [0] and [4] to [2]
        lastFrame1 = frame_buffer[0]
        lastFrame3 = frame_buffer[2]

        # the FD5_mask() function is returning a mask for lastFrame3, so need to input 
        # an adjusted frame number
        mask_frame_num = frame_count - 2

        # compute motion mask and save to file
        FD5_mask(lastFrame1, lastFrame3, currentFrame, video_name, mask_frame_num)

    # close the video file immediately after processing is finished
    cap.release()

def process_video_directory_masks(video_dir: Path):
    # this function scans the given directory for mp4 files and calls the motion mask function on them
    if not video_dir.exists():
        print(f"Directory not found: {video_dir}")
        return
    
    # extract list of all mp4 files in directory
    video_files = list(video_dir.glob("*.mp4"))

    if not video_files:
        print(f"No .mp4 files found in {video_dir}")
        return
    
    # archive: single-threaded implementation
    # for video_path in video_files:
    #     process_video_masks(video_path)

    # parallelised implementation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_video_masks, video_files)

def process_single_video_masks(video_path: Path):
    """
    THIS FUNCTION IS JUST FOR TESTING PURPOSES
    It only processes a single video and calls the motion mask function on it
    """
    if not video_path.exists():
        print(f"File not found: {video_path}")
        return
        
    print(f"Processing single video: {video_path.name}")
    
    process_video_masks(video_path)

if __name__ == "__main__":
    # # To test code on a single video first
    # test_video_path = TRAIN_VIDEOS_DIR / "phantom09.mp4"
    # print("--- Testing Single Video ---")
    # process_single_video_masks(test_video_path)
    
    print("--- Processing Training Videos ---")
    process_video_directory_masks(TRAIN_VIDEOS_DIR)
    print("\n--- Processing Test Videos ---")
    process_video_directory_masks(TEST_VIDEOS_DIR)