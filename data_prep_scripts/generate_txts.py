# this script generates the .txt files required for YOLO training
# every frame must be listed in a .txt file
# the labels/RGB frames/motion mask frames must match
# this script checks for three-way match before writing the frame to .txt

import os
import sys
from pathlib import Path
from typing import Literal
# connect to config by adding the parent directory of the current working directory 
# to the list of paths where Python searches for modules
sys.path.append('..')
# Import paths from config
from config import (
    PROCESSED_DATA_DIR,
    IMAGES_TRAIN_DIR, MASKS_TRAIN_DIR, LABELS_TRAIN_DIR,
    IMAGES_VAL_DIR, MASKS_VAL_DIR, LABELS_VAL_DIR,
    IMAGES_TEST_DIR, MASKS_TEST_DIR, LABELS_TEST_DIR
)

def generate_txt(
        split_name: Literal["train", "val", "test"],
        rgb_dir, 
        mask_dir,
        label_dir,
        output_dir
    ):
    """
    This function generates .txt files for bimodal YOLO training
    It enforces a three-way match (RGB frame, Mask frame, Label)
    """    
    rgb_dir = Path(rgb_dir)
    mask_dir = Path(mask_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)

    # YOLO convention is to store the .txt files at the dataset root folder
    rgb_txt_path = output_dir / f"{split_name}.txt"
    mask_txt_path = output_dir / f"{split_name}2.txt"

    # sort the images to ensure deterministic output
    rgb_images = sorted([file for file in os.listdir(rgb_dir) if file.endswith((".jpg"))])
    matched_count = 0

    with open(rgb_txt_path, "w") as file_rgb, open(mask_txt_path, "w") as file_mask:
        for rgb_img in rgb_images:
            base_name = os.path.splitext(rgb_img)[0]
            
            # construct paths for the expected mask and label
            mask_path = mask_dir / f"{base_name}.jpg"
            label_path = label_dir / f"{base_name}.txt"
            rgb_path = rgb_dir / rgb_img
            
            # check for three-way match
            if mask_path.exists() and label_path.exists():
                file_rgb.write(f"{rgb_path.resolve()}\n")
                file_mask.write(f"{mask_path.resolve()}\n")
                matched_count += 1
            else:
                if not mask_path.exists():
                    print(f"[{split_name.upper()}] Missing mask for {rgb_img}. Skipping.")
                if not label_path.exists():
                    print(f"[{split_name.upper()}] Missing label for {rgb_img}. Skipping.")

    print(f"[{split_name.upper()}] complete. Wrote {matched_count} three-way matches to:")
    print(f" - {rgb_txt_path}")
    print(f" - {mask_txt_path}\n")

if __name__ == "__main__":
    # YOLO convention is to store the .txt files at the dataset root folder
    dataset_root = Path(PROCESSED_DATA_DIR)
    
    splits = [
        ('train', IMAGES_TRAIN_DIR, MASKS_TRAIN_DIR, LABELS_TRAIN_DIR),
        ('val', IMAGES_VAL_DIR, MASKS_VAL_DIR, LABELS_VAL_DIR),
        ('test', IMAGES_TEST_DIR, MASKS_TEST_DIR, LABELS_TEST_DIR)
    ]

    for split_name, rgb_dir, mask_dir, label_dir in splits:
        # Check if the directories exist before processing to avoid errors on missing splits
        if Path(rgb_dir).exists():
            generate_txt(split_name, rgb_dir, mask_dir, label_dir, dataset_root)
        else:
            print(f"Skipping '{split_name}' split: Directory {rgb_dir} does not exist.")