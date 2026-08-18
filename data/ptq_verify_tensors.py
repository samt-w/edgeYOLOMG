import sys
import cv2
import numpy as np
import random
from pathlib import Path

# add repo root filepath dynamically to import modules from other directories
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import PROJECT_ROOT

RGB_DIR = PROJECT_ROOT / "data/ptq_calibration_RGB_tensors_1280"
MASK_DIR = PROJECT_ROOT / "data/ptq_calibration_MASKS_tensors_1280"

def npy_to_image(npy_path):
    """
    reverses the PyTorch formatting to return a viewable OpenCV BGR image
    """
    # load the numpy array
    tensor = np.load(npy_path)
    
    # drop the batch dimension: (1, 3, 640, 640) -> (3, 640, 640)
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
        
    # transpose from CHW to HWC: (3, 640, 640) -> (640, 640, 3)
    img_array = tensor.transpose(1, 2, 0)
    
    # denormalise from 0.0-1.0 float to 0-255 uint8
    img_array = (img_array * 255.0).clip(0, 255).astype(np.uint8)
    
    # convert RGB back to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_bgr

def main():
    if not RGB_DIR.exists() or not MASK_DIR.exists():
        print("Error: Calibration directories not found.")
        return

    # get all .npy files
    rgb_files = list(RGB_DIR.glob("*_rgb.npy"))
    
    if not rgb_files:
        print("No .npy files found in the calibration directories.")
        return

    print(f"Found {len(rgb_files)} tensor pairs. Press 'n' for next, 'q' to quit.")

    # shuffle for random inspection
    random.shuffle(rgb_files)

    for rgb_path in rgb_files:
        mask_filename = rgb_path.name.replace("_rgb.npy", "_mask.npy")
        mask_path = MASK_DIR / mask_filename
        
        if not mask_path.exists():
            print(f"Missing matching mask for {rgb_path.name}")
            continue

        # reconstruct images
        img_view = npy_to_image(rgb_path)
        mask_view = npy_to_image(mask_path)

        # concatenate images side-by-side
        combined_view = cv2.hconcat([img_view, mask_view])

        # display
        window_title = f"Verification: {rgb_path.stem.replace('_rgb', '')} | Left: RGB, Right: Mask"
        cv2.imshow(window_title, combined_view)
        
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == ord('q'):
            print("Quitting verification.")
            break
        elif key == ord('n'):
            continue

if __name__ == "__main__":
    main()