import os
import concurrent.futures
import itertools
import shutil
import xml.etree.ElementTree as ET
import random
import sys
import cv2

# connect to config by adding the parent directory of the current working directory 
# to the list of paths where Python searches for modules
sys.path.append('..')
from config import (
    RANDOM_SEED, 
    ANNOTATIONS_DIR,
    IMAGES_DIR, MASKS_DIR, 
    TRAIN_DESTS, VAL_DESTS, TEST_DESTS,
    IMAGES_TRAIN_DIR, MASKS_TRAIN_DIR, LABELS_TRAIN_DIR,
    IMAGES_VAL_DIR, MASKS_VAL_DIR, LABELS_VAL_DIR,
    IMAGES_TEST_DIR, MASKS_TEST_DIR, LABELS_TEST_DIR,
    IMAGES_1280_TRAIN_DIR, MASKS_1280_TRAIN_DIR, LABELS_1280_TRAIN_DIR,
    IMAGES_1280_VAL_DIR, MASKS_1280_VAL_DIR, LABELS_1280_VAL_DIR,
    IMAGES_1280_TEST_DIR, MASKS_1280_TEST_DIR, LABELS_1280_TEST_DIR,
    IMAGES_640_TRAIN_DIR, MASKS_640_TRAIN_DIR, LABELS_640_TRAIN_DIR,
    IMAGES_640_VAL_DIR, MASKS_640_VAL_DIR, LABELS_640_VAL_DIR,
    IMAGES_640_TEST_DIR, MASKS_640_TEST_DIR, LABELS_640_TEST_DIR,
    ARD100_TRAIN_LIST, ARD100_TEST_LIST 
)

# import YOLOv5 letterbox for image resizing
from utils.augmentations import letterbox

# define validation split
random.seed(RANDOM_SEED)
VALIDATION_SPLIT_RATIO = 0.20
train_videos_shuffled = ARD100_TRAIN_LIST.copy()
random.shuffle(train_videos_shuffled)
val_split_index = int(len(train_videos_shuffled) * (1 - VALIDATION_SPLIT_RATIO))

train_videos = train_videos_shuffled[:val_split_index]
val_videos = train_videos_shuffled[val_split_index:]
test_videos = ARD100_TEST_LIST

def ensure_dirs():
    """Create all required output directories defined in config."""
    for directory in [
        IMAGES_TRAIN_DIR, MASKS_TRAIN_DIR, LABELS_TRAIN_DIR,
        IMAGES_VAL_DIR, MASKS_VAL_DIR, LABELS_VAL_DIR,
        IMAGES_TEST_DIR, MASKS_TEST_DIR, LABELS_TEST_DIR,
        IMAGES_1280_TRAIN_DIR, MASKS_1280_TRAIN_DIR, LABELS_1280_TRAIN_DIR,
        IMAGES_1280_VAL_DIR, MASKS_1280_VAL_DIR, LABELS_1280_VAL_DIR,
        IMAGES_1280_TEST_DIR, MASKS_1280_TEST_DIR, LABELS_1280_TEST_DIR,
        IMAGES_640_TRAIN_DIR, MASKS_640_TRAIN_DIR, LABELS_640_TRAIN_DIR,
        IMAGES_640_VAL_DIR, MASKS_640_VAL_DIR, LABELS_640_VAL_DIR,
        IMAGES_640_TEST_DIR, MASKS_640_TEST_DIR, LABELS_640_TEST_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)

def convert_to_yolo_format(size, box):
    """Converts PASCAL VOC bounding box to YOLO format (normalized xywh)."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    # Calculate center x, center y, width, and height
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    
    # Normalize
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_to_padded_yolo_format(box, ratio, pad, new_size):
    """Adjusts PASCAL VOC box for letterbox padding and converts to YOLO format."""
    # scale and pad the bounding box coordinates
    new_xmin = box[0] * ratio[0] + pad[0]
    new_xmax = box[1] * ratio[0] + pad[0]
    new_ymin = box[2] * ratio[1] + pad[1]
    new_ymax = box[3] * ratio[1] + pad[1]
    
    # convert to normalized YOLO format against the new square size
    x_center = ((new_xmin + new_xmax) / 2.0) / new_size
    y_center = ((new_ymin + new_ymax) / 2.0) / new_size
    w = (new_xmax - new_xmin) / new_size
    h = (new_ymax - new_ymin) / new_size
    return (x_center, y_center, w, h)

def process_single_video(video_id, dest_paths):
    """Processes a video, converts annotations, and copies files to target directories, and cleans up."""
    small_num = 0
    total_processed = 0
    # Define raw paths based on standard structure
    # Update these if the raw structure differs from the original script's assumptions
    img_dir = IMAGES_DIR / video_id
    mask_dir = MASKS_DIR / video_id
    anno_dir = ANNOTATIONS_DIR / video_id
    
    if not anno_dir.exists():
        print(f"Warning: Annotation directory missing for {video_id}, skipping.")
        return 0, 0

    # unpack destinations for each image resolution
    orig_dest, dest_1280, dest_640 = dest_paths['orig'], dest_paths['1280'], dest_paths['640']

    # iterate over all XML files rather than hardcoded index matching
    for xml_file in os.listdir(anno_dir):
        if not xml_file.endswith('.xml'):
            continue
            
        xml_path = anno_dir / xml_file
        base_name = os.path.splitext(xml_file)[0]
        img_name = f"{base_name}.jpg"
        
        img_path = img_dir / img_name
        mask_path = mask_dir / img_name
        
        if not img_path.exists():
            continue
            
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        if root.find('object') is None:
            continue
            
        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)
        
        # extract raw bounding boxes from XML
        raw_boxes = []
        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            b1 = float(xmlbox.find('xmin').text)
            b2 = float(xmlbox.find('xmax').text)
            b3 = float(xmlbox.find('ymin').text)
            b4 = float(xmlbox.find('ymax').text)
            area = (b2 - b1) * (b4 - b3)
            
            # minimum area threshold from original script
            if area >= 25:
                raw_boxes.append((b1, b2, b3, b4))
            else:
                small_num += 1
        
        if not raw_boxes:
            continue # skip if no valid boxes

        # process images at original resolution
        orig_labels = []
        for box in raw_boxes:
            yolo_bbox = convert_to_yolo_format((img_w, img_h), box)
            orig_labels.append(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")
        
        with open(orig_dest[2] / f"{base_name}.txt", 'w') as file:
            file.write('\n'.join(orig_labels) + '\n')
            
        shutil.copy(img_path, orig_dest[0] / img_name)
        if mask_path.exists():
            shutil.copy(mask_path, orig_dest[1] / img_name)

        # load images for resizing at 1280 and 640 resolution
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path)) if mask_path.exists() else None

        # process 1280 resolution
        img_1280, ratio_1280, pad_1280 = letterbox(img, new_shape=(1280, 1280), auto=False)
        cv2.imwrite(str(dest_1280[0] / img_name), img_1280)
        if mask is not None:
            # setting color=(0, 0, 0) so the padding is black (i.e. "no motion") for the motion mask
            mask_1280, _, _ = letterbox(mask, new_shape=(1280, 1280), auto=False, color=(0, 0, 0))
            cv2.imwrite(str(dest_1280[1] / img_name), mask_1280)

        labels_1280 = []
        for box in raw_boxes:
            yolo_bbox = convert_to_padded_yolo_format(box, ratio_1280, pad_1280, 1280)
            labels_1280.append(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")
        
        with open(dest_1280[2] / f"{base_name}.txt", 'w') as f:
            f.write('\n'.join(labels_1280) + '\n')

        # process 640 resolution
        img_640, ratio_640, pad_640 = letterbox(img, new_shape=(640, 640), auto=False)
        cv2.imwrite(str(dest_640[0] / img_name), img_640)
        if mask is not None:
            # setting color=(0, 0, 0) so the padding is black (i.e. "no motion") for the motion mask
            mask_640, _, _ = letterbox(mask, new_shape=(640, 640), auto=False, color=(0, 0, 0))
            cv2.imwrite(str(dest_640[1] / img_name), mask_640)

        labels_640 = []
        for box in raw_boxes:
            yolo_bbox = convert_to_padded_yolo_format(box, ratio_640, pad_640, 640)
            labels_640.append(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")
        
        with open(dest_640[2] / f"{base_name}.txt", 'w') as f:
            f.write('\n'.join(labels_640) + '\n')

        total_processed += 1

    # delete intermediate folders created by extract_frames.py and generate_motion_masks.py to save disk space
    if img_dir.exists():
        shutil.rmtree(img_dir)
    if mask_dir.exists():
        shutil.rmtree(mask_dir)

    return total_processed, small_num

def process_dataset_split(video_list, dest_paths):
    """Distributes video processing across multiple threads - as this is an I/O-bound task."""
    total_processed = 0
    total_small = 0
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # itertools.repeat ensures the destination paths are passed to every function call
        results = executor.map(
            process_single_video, 
            video_list,
            itertools.repeat(dest_paths)
        )

    # aggregate results from all processed videos
    for proc, small in results:
        total_processed += proc
        total_small += small
        
    return total_processed, total_small

if __name__ == "__main__":
    ensure_dirs()
    
    print("Processing Training Set...")
    train_proc, train_small = process_dataset_split(train_videos, TRAIN_DESTS)
    
    print("Processing Validation Set...")
    val_proc, val_small = process_dataset_split(val_videos, VAL_DESTS)
    
    print("Processing Test Set...")
    test_proc, test_small = process_dataset_split(test_videos, TEST_DESTS)
    
    print("\nDataset Generation Complete.")
    print(f"Train samples: {train_proc}")
    print(f"Val samples:   {val_proc}")
    print(f"Test samples:  {test_proc}")
    print(f"Total small objects ignored (<25 pixels): {train_small + val_small + test_small}")