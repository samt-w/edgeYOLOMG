"""
Stratified sampling script to extract the calibration dataset for TensorRT 
post-training quantisation

Extracts exactly 512 frames from ARD100 training set:
- 18 Multiple Object frames (all the frames of this class which were in the training set)
- 75 None frames
- 419 Single Object frames

The first 2 and last 2 frames of every video clip are excluded to align
with 5-frame buffer motion mask constraints
"""

import sys
from pathlib import Path

# add repo root filepath dynamically to import modules from other directories
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import shutil
import random
import concurrent.futures
import xml.etree.ElementTree as ET
from config import (
    ANNOTATIONS_DIR,
    IMAGES_TRAIN_DIR, MASKS_TRAIN_DIR,
    IMAGES_VAL_DIR, MASKS_VAL_DIR,
    IMAGES_TEST_DIR, MASKS_TEST_DIR,
    ARD100_TRAIN_LIST,
    RECOMMENDED_CORES,
    RANDOM_SEED,
    PROJECT_ROOT,
)

# set random seed for reproducibility
random.seed(RANDOM_SEED)

# output directory definitions
OUTPUT_RGB_DIR = PROJECT_ROOT / "data/ptq_calibration_RGB"
OUTPUT_MASK_DIR = PROJECT_ROOT / "data/ptq_calibration_MASKS"


def extract_frame_number(filepath):
    """
    Extract integer frame number from filename (e.g., 'phantom09_0123.xml' -> 123).
    """
    try:
        return int(filepath.stem.split("_")[-1])
    except (IndexError, ValueError):
        return 0

def find_processed_files(stem):
    """
    Searches across Train, Val, and Test directories for the processed image and mask
    """
    search_dirs = [
        (IMAGES_TRAIN_DIR, MASKS_TRAIN_DIR),
        (IMAGES_VAL_DIR, MASKS_VAL_DIR),
        (IMAGES_TEST_DIR, MASKS_TEST_DIR)
    ]
    
    for img_dir, mask_dir in search_dirs:
        img_path = img_dir / f"{stem}.jpg"
        if img_path.exists():
            mask_path = mask_dir / f"{stem}.jpg"
            return img_path, mask_path if mask_path.exists() else None
            
    return None, None

def process_single_video(video_name):
    """
    Scans training video, applies [2:-2] boundary filters, and categorises frames by class.
    """
    video_annot_dir = ANNOTATIONS_DIR / video_name
    local_candidates = {"multiple": [], "none": [], "single": []}

    if not video_annot_dir.is_dir():
        return video_name, local_candidates

    # gather all XML files and sort numerically by frame index
    xml_files = [f for f in video_annot_dir.iterdir() if f.suffix.lower() == ".xml"]
    xml_files.sort(key = extract_frame_number)

    # exclude first 2 and last 2 frames
    if len(xml_files) <= 4:
        return video_name, local_candidates

    valid_xml_files = xml_files[2:-2]

    # parse XML and classify candidates
    for xml_path in valid_xml_files:
        stem = xml_path.stem
        # look for the file across all splits
        img_path, mask_path = find_processed_files(stem)

        # if the image was skipped during dataset creation, skip it here
        if not img_path:
            continue

        try:
            tree = ET.parse(xml_path)
            num_objects = len(list(tree.iter("object")))

            entry = {
                "stem": stem,
                "img_path": img_path,
                "mask_path": mask_path
            }

            if num_objects > 1:
                local_candidates["multiple"].append(entry)
            elif num_objects == 0:
                local_candidates["none"].append(entry)
            else:
                local_candidates["single"].append(entry)

        except ET.ParseError:
            continue

    return video_name, local_candidates

def collect_valid_candidates_multiprocess():
    """
    Distributes frame extraction across CPU
    """
    candidates = {"multiple": [], "none": [], "single": []}
    total_videos = len(ARD100_TRAIN_LIST)

    print(f"Scanning {total_videos} training videos...")

    completed_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers = RECOMMENDED_CORES) as executor:
        futures = {
            executor.submit(process_single_video, vid): vid
            for vid in ARD100_TRAIN_LIST
        }

        for future in concurrent.futures.as_completed(futures):
            video_name, local_candidates = future.result()
            completed_count += 1

            for cat in ["multiple", "none", "single"]:
                candidates[cat].extend(local_candidates[cat])

            num_found = (
                len(local_candidates["multiple"])
                + len(local_candidates["none"])
                + len(local_candidates["single"])
            )
            print(
                f"[{completed_count:>2}/{total_videos}] Processed {video_name:<12} | "
                f"Valid frames: {num_found:>5} (Multi: {len(local_candidates['multiple'])}, "
                f"None: {len(local_candidates['none'])}, Single: {len(local_candidates['single'])})"
            )

    return candidates

def sample_and_copy():
    # create destination folders
    OUTPUT_RGB_DIR.mkdir(parents = True, exist_ok = True)
    OUTPUT_MASK_DIR.mkdir(parents = True, exist_ok = True)

    candidates = collect_valid_candidates_multiprocess()

    num_multiple_avail = len(candidates["multiple"])
    num_none_avail = len(candidates["none"])
    num_single_avail = len(candidates["single"])

    print("-----------------------------------------------")
    print("Candidate Frames Summary:")
    print(f"- Multiple Objects Frames available: {num_multiple_avail}")
    print(f"- None (Background) Frames available: {num_none_avail}")
    print(f"- Single Object Frames available: {num_single_avail}")

    # specify target frame numbers
    target_multiple = min(18, num_multiple_avail)

    # sample all available multiple objects (up to 18)
    selected_multiple = candidates["multiple"][:target_multiple]

    # sample 75 None frames randomly
    target_none = min(75, num_none_avail)
    selected_none = random.sample(candidates["none"], target_none)

    # sample remainder of 512 frames with Single Object frames randomly
    target_single = 512 - len(selected_multiple) - len(selected_none)
    if num_single_avail < target_single:
        raise ValueError(f"Not enough 'Single' frames available to meet 512 target: only ({num_single_avail} available).")
    selected_single = random.sample(candidates["single"], target_single)

    selected_dataset = selected_multiple + selected_none + selected_single

    print("-----------------------------------------------")
    print(f"Sampling Breakdown (Total: {len(selected_dataset)} frames):")
    print(f"- Selected Multiple: {len(selected_multiple)}")
    print(f"- Selected None: {len(selected_none)}")
    print(f"- Selected Single: {len(selected_single)}")

    # copy files
    print("-----------------------------------------------")
    print(f"Copying RGB images to: {OUTPUT_RGB_DIR.resolve()}")
    print(f"Copying Motion Masks to: {OUTPUT_MASK_DIR.resolve()}")

    masks_copied = 0
    total_to_copy = len(selected_dataset)

    for idx, item in enumerate(selected_dataset, 1):
        dst_img = OUTPUT_RGB_DIR / f"{item['stem']}.jpg"
        shutil.copy2(item["img_path"], dst_img)

        # copy matching mask if available
        if item["mask_path"]:
            dst_mask = OUTPUT_MASK_DIR / f"{item['stem']}.jpg"
            shutil.copy2(item["mask_path"], dst_mask)
            masks_copied += 1

        if idx % 50 == 0 or idx == total_to_copy:
            print(f"Progress: [{idx:>3}/{total_to_copy}] files copied...")

    print("-----------------------------------------------")
    print(f"Complete: Successfully exported {total_to_copy} RGB images")
    print(f"and {masks_copied} motion masks to calibration directories.")

if __name__ == "__main__":
    sample_and_copy()