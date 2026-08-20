"""
this script acts as a dataloader for the Polygraphy module

it is designed to load the first YOLOMG pair of inputs (RGB and masks) for Polygraphy to use
(e.g. in ```polygraphy debug precision```) from the given data directory
"""
import numpy as np
from pathlib import Path

def load_data():
    rgb_files = sorted(Path("data/ptq_calibration_RGB_tensors_640").glob("*_rgb.npy"))
    for rgb_path in rgb_files:
        mask_path = Path("data/ptq_calibration_MASKS_tensors_640") / rgb_path.name.replace("_rgb.npy", "_mask.npy")
        yield {
            "images": np.load(rgb_path).astype(np.float32),
            "masks": np.load(mask_path).astype(np.float32),
        }
        # end the script (and the dataloader) after the first iteration
        break

if __name__ == "__main__":
    print("Testing dataloader...\n")
    
    # initialise the generator
    data_gen = load_data()
    
    # fetch the first batch
    first_batch = next(data_gen)
    
    # print output stats
    for input_name, tensor in first_batch.items():
        print(f"Input: '{input_name}'")
        print(f"- Type:  {type(tensor)}")
        print(f"- Shape: {tensor.shape}")
        print(f"- Dtype: {tensor.dtype}")
        print(f"- Min/Max: {tensor.min():.3f} / {tensor.max():.3f}\n")