import cv2
import sys
from collections import deque
from pathlib import Path
from MOD_Functions import motion_compensate
import concurrent.futures

# connect to config by adding the parent directory of the current working directory 
# to the list of paths where Python searches for modules
sys.path.append('..')
from config import MASKS_DIR, TRAIN_VIDEOS_DIR, TEST_VIDEOS_DIR

def FD5_mask(lastFrame1, lastFrame2, currentFrame, video_name, frame_count):
    """
    This function computes a motion mask for a particular frame, lastFrame2, given one preceding frame,
    lastFrame1, and and one succeeding frame, currentFrame.

    The temporal order of the frames is:
    - lastFrame1 (oldest)
    - lastFrame2
    - currentFrame (newest)

    This naming convention comes from the original code and is somewhat confusing - the frame_count
    variable must be set to the frame count for lastFrame2, not currentFrame, as the motion mask belongs
    to lastFrame2
    """
    # Blur images to eliminate some low-level noise 
    # Convert to grayscale to reduce computational complexity
    lastFrame1 = cv2.cvtColor(cv2.GaussianBlur(lastFrame1, (11, 11), 0), cv2.COLOR_BGR2GRAY)
    lastFrame2 = cv2.cvtColor(cv2.GaussianBlur(lastFrame2, (11, 11), 0), cv2.COLOR_BGR2GRAY)
    currentFrame = cv2.cvtColor(cv2.GaussianBlur(currentFrame, (11, 11), 0), cv2.COLOR_BGR2GRAY)

    # Compute motion compensation and differences
    img_compensate1, mask1, avg_dist1, motion_x1, motion_y1, homo_matrix = motion_compensate(lastFrame1, lastFrame2)
    frameDiff1 = cv2.absdiff(lastFrame2, img_compensate1)

    img_compensate2, mask2, avg_dist2, motion_x2, motion_y2, homo_matrix2 = motion_compensate(currentFrame, lastFrame2)
    frameDiff2 = cv2.absdiff(lastFrame2, img_compensate2)

    # Average the differences to create the continuous motion mask
    # original code was: frameDiff = (frameDiff1 + frameDiff2) / 2
    # but this returned a warning about datatypes from OpenCV, so the following amendment prevents numerical overflow
    frameDiff = cv2.addWeighted(frameDiff1, 0.5, frameDiff2, 0.5, 0) 

    # Construct the save path using pathlib
    save_dir = MASKS_DIR / video_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the output image
    filename = f"{video_name}_{str(frame_count).zfill(4)}.jpg"
    cv2.imwrite(str(save_dir / filename), frameDiff)

    return 0

def process_video_masks(video_path: Path):
    """
    This function processes frames and generate masks for a single video
    """
    video_name = video_path.stem
    print(f"Processing motion masks for: {video_name}")

    # for debugging purposes, log moment processing starts
    with open("mask_telemetry.log", "a") as f:
        f.write(f"STARTED: {video_name}\n")
    # ----------------------------------------------------

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

        # for debugging purposes, log every 500 frames
        if frame_count % 500 == 0:
            with open("mask_telemetry.log", "a") as f:
                f.write(f"PROGRESS: {video_name} at frame {frame_count}\n")
        # ----------------------------------------------------

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

    # for debugging purposes, log moment processing ends
    with open("mask_telemetry.log", "a") as f:
        f.write(f"FINISHED: {video_name} (Total frames: {frame_count})\n")
    # ----------------------------------------------------

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