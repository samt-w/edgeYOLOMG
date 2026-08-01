"""
this script takes trained YOLOMG .pt weights and uses them to 
detect drones in video input. It measures the performance of the detection
using mAP and FPS.
"""

# read video
# allow buffer to be initialised (to warm up GPU and for motion masks)
# begin timer
# extract frame
# compute motion mask for each frame
# make prediction
# end timer
# discard frame in buffer
# report accuracy and frame time
# convert accuracy and frame time to mAP and FPS

import os
import sys
from pathlib import Path

# add repo root filepath dynamically to import modules from other directories
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2
from data_prep_scripts.MOD_Functions import motion_compensate

def compute_mask_in_memory(lastFrame1, lastFrame2, currentFrame):
    """
    This function computes a motion mask for a particular frame, lastFrame2, given one preceding frame,
    lastFrame1, and and one succeeding frame, currentFrame.

    The temporal order of the frames is:
    - lastFrame1 (oldest)
    - lastFrame2
    - currentFrame (newest)

    This naming convention comes from the original code.

    This function differs to the data preparation counterpart because it holds the 
    mask in memory rather than saving it to disk. This is necessary for inference.
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
    frameDiff = cv2.addWeighted(frameDiff1, 0.5, frameDiff2, 0.5, 0) 
    
    # Inflate 2D motion mask to 3D
    frameDiff_3c = cv2.cvtColor(frameDiff, cv2.COLOR_GRAY2BGR)
    return frameDiff_3c

if __name__ == "__main__":
    pass