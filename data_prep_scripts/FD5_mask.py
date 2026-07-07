import cv2
import sys
import numpy as np
from MOD_Functions import motion_compensate

# connect to global paths/variables
sys.path.append('..')
from config import ARD100_MASKS_DIR

def FD5_mask(lastFrame1, lastFrame2, currentFrame, video_name, frame_count):
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
    frameDiff = (frameDiff1 + frameDiff2) / 2

    # Construct the save path using pathlib
    save_dir = MASKS_DIR / video_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the output image
    filename = f"{video_name}_{str(frame_count).zfill(4)}.jpg"
    cv2.imwrite(str(save_dir / filename), frameDiff)

    return 0