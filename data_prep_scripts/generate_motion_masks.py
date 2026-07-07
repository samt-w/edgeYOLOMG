import cv2
import sys
from collections import deque
from pathlib import Path
from motion_mask import FD5_mask
from config import TRAIN_VIDEOS_DIR, TEST_VIDEOS_DIR

# connect to global paths/variables
sys.path.append('..')

def process_video_directory(video_dir: Path):
    # this function scans directory for mp4 files and computes motion masks for them
    if not video_dir.exists():
        print(f"Directory not found: {video_dir}")
        return
    
    # extract list of all mp4 files in directory
    video_files = list(video_dir.glob("*.mp4"))

    if not video_files:
        print(f"No .mp4 files found in {video_dir}")
        return
    
    for video_path in video_files:
        video_name = video_path.stem
        print(f"Processing: {video_name}")

        # create an OpenCV VideoCapture object
        cap = cv2.VideoCapture(str(video_path))

        # use Python's deque function to manage the frame buffer
        frame_buffer = deque(maxlen = 5)
        frame_count = 0

        # while the capture is initialised 
        while cap.isOpened():
            # when the loop reaches the end of the video, return_status is False and the loop breaks
            return_status, currentFrame = cap.read()
            if not return_status:
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

    # close the video file once have iterated through all frames
    cap.release()

if __name__ == "__main__":
    print("--- Processing Training Videos ---")
    process_video_directory(TRAIN_VIDEOS_DIR)
    
    print("\n--- Processing Test Videos ---")
    process_video_directory(TEST_VIDEOS_DIR)