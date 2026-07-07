import cv2
import sys
from MOD_Functions import motion_compensate

# connect to global paths/variables
sys.path.append('..')
from config import MASKS_DIR

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
    frameDiff = (frameDiff1 + frameDiff2) / 2

    # Construct the save path using pathlib
    save_dir = MASKS_DIR / video_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the output image
    filename = f"{video_name}_{str(frame_count).zfill(4)}.jpg"
    cv2.imwrite(str(save_dir / filename), frameDiff)

    return 0