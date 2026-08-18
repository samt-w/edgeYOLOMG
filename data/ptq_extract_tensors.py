"""
Stratified sampling script to extract the calibration dataset AS NUMPY TENSORS for TensorRT 
post-training quantisation

Extracts exactly 512 frames from ARD100 training set:
- 18 Multiple Object frames (all the frames of this class which were in the training set)
- 75 None frames
- 419 Single Object frames

It then preprocesses these using the same CPU-based OpenCV functions, and saves .npy tensors
"""

import sys
from pathlib import Path
import random
import concurrent.futures
import xml.etree.ElementTree as ET
import cv2
import torch
import numpy as np

# add repo root filepath dynamically to import modules from other directories
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import (
    ANNOTATIONS_DIR,
    TRAIN_VIDEOS_DIR,
    ARD100_TRAIN_LIST,
    RECOMMENDED_CORES,
    RANDOM_SEED,
    PROJECT_ROOT,
)

random.seed(RANDOM_SEED)

def extract_frame_number(filepath):
    """
    Extract integer frame number from filename (e.g., 'phantom09_0123.xml' -> 123).
    """
    try:
        return int(filepath.stem.split("_")[-1])
    except (IndexError, ValueError):
        return 0

def process_single_video_xml(video_name):
    """
    Scans training video annotations and categorises frames by class
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
        frame_idx = extract_frame_number(xml_path)
        
        try:
            tree = ET.parse(xml_path)
            num_objects = len(list(tree.iter("object")))

            entry = {"stem": stem, "frame_idx": frame_idx}

            if num_objects > 1:
                local_candidates["multiple"].append(entry)
            elif num_objects == 0:
                local_candidates["none"].append(entry)
            else:
                local_candidates["single"].append(entry)
        except ET.ParseError:
            continue

    return video_name, local_candidates

def candidate_sort_key(item):
    """
    helper function for candidate frame sorting
    """
    return (item["video_name"], item["frame_idx"])

def collect_valid_candidates():
    """
    parallelised processing of the frame annotation extraction
    """
    candidates = {"multiple": [], "none": [], "single": []}
    total_videos = len(ARD100_TRAIN_LIST)
    print(f"Scanning XMLs for {total_videos} training videos...")

    with concurrent.futures.ProcessPoolExecutor(max_workers = RECOMMENDED_CORES) as executor:
        futures = {executor.submit(process_single_video_xml, vid): vid for vid in ARD100_TRAIN_LIST}
        for future in concurrent.futures.as_completed(futures):
            video_name, local_candidates = future.result()
            for cat in ["multiple", "none", "single"]:
                for item in local_candidates[cat]:
                    item["video_name"] = video_name
                    item["class"] = cat
                candidates[cat].extend(local_candidates[cat])

    # sort so that random.sample() is reproducible regardless of worker completion ordering
    for cat in candidates:
        candidates[cat].sort(key = candidate_sort_key)

    return candidates

def sample_and_group_frames():
    candidates = collect_valid_candidates()

    target_multiple = min(18, len(candidates["multiple"]))
    selected_multiple = random.sample(candidates["multiple"], target_multiple)

    target_none = min(75, len(candidates["none"]))
    selected_none = random.sample(candidates["none"], target_none)

    target_single = 512 - len(selected_multiple) - len(selected_none)
    selected_single = random.sample(candidates["single"], target_single)

    selected_dataset = selected_multiple + selected_none + selected_single

    frames_by_video = {}
    for item in selected_dataset:
        vid = item["video_name"]
        if vid not in frames_by_video:
            frames_by_video[vid] = {}
        frames_by_video[vid][item["frame_idx"]] = item["class"]

    print("-----------------------------------------------")
    print(f"Sampled {len(selected_dataset)} frames across {len(frames_by_video)} videos.")
    return frames_by_video

def letterbox_cpu(
        im,
        new_shape = (1280, 1280),
        color = (114, 114, 114),
        auto = True,
        scaleFill = False,
        scaleup = True,
        stride = 32
    ):
    """
    CPU equivalent of letterbox_cuda
    """
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup: # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto: # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride) # wh padding
    # if scaleFill is true, no padding is applied
    elif scaleFill: # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
    
    # compute padding size
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad: # resize
        im = cv2.resize(im, new_unpad, interpolation = cv2.INTER_LINEAR)
    
    # compute which sides should be padded
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # perform padding
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im

def motion_compensate_cpu(frame1, frame2, ones_mask, pts_prev):
    """
    CPU equivalent of motion_compensate_cuda
    """
    width, height = frame2.shape[1], frame2.shape[0]
    scale = 2
    
    frame1_grid = cv2.resize(frame1, (960 * scale, 540 * scale), interpolation=cv2.INTER_CUBIC)
    frame2_grid = cv2.resize(frame2, (960 * scale, 540 * scale), interpolation=cv2.INTER_CUBIC)

    # use the Lucas-Kanade algorithm to track how grid points moved between frames
    pts_cur, status, err = cv2.calcOpticalFlowPyrLK(frame1_grid, frame2_grid, pts_prev, None, winSize=(15, 15), maxLevel=3)

    # Select the good points
    good_new = pts_cur[status.flatten() == 1].reshape(-1, 2) # Tracking points in the current frame
    good_old = pts_prev[status.flatten() == 1].reshape(-1, 2) # Tracking point in the previous frame

    # vectorised operations for computing Euclidean distance for points. If point moved more than a given
    # threshold (50px for 1920px images), remove it (to avoid tracking errors)
    if len(good_new) > 0 and len(good_old) > 0:
        # compute difference between points
        diff = good_new - good_old
        motion_distance = np.linalg.norm(diff, axis=1)
        valid_mask = motion_distance <= 50
        
        filtered_new = good_new[valid_mask]
        filtered_old = good_old[valid_mask]
    else:
        filtered_new = np.array([])
        filtered_old = np.array([])

    # if fewer than 15 valid points, just use identity matrix as not enough data
    # otherwise, use the RANSAC algorithm to compute the transformation matrix
    if len(filtered_old) < 15:
        homography_matrix = np.array([[0.999, 0, 0], [0, 0.999, 0], [0, 0, 1]], dtype=np.float32)
    else:
        homography_matrix, _ = cv2.findHomography(filtered_new, filtered_old, cv2.RANSAC, 3.0)
        # ensure any RANSAC failures are caught
        if homography_matrix is None:
            homography_matrix = np.array([[0.999, 0, 0], [0, 0.999, 0], [0, 0, 1]], dtype=np.float32)

    # Calculate the transformed image based on the transformation matrix
    compensated = cv2.warpPerspective(frame1, homography_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    # Calculate mask of valid pixels (invalid pixels are marked as 255)
    # AMENDMENT TO ORIGINAL CODE:
    # instead of computing inverse matrix, just warp a uniform matrix using the transformation matrix
    # invalid pixels get marked as 0, and then get inverted using .bitwise_not() to match original logic
    warped_ones = cv2.warpPerspective(ones_mask, homography_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    mask = cv2.bitwise_not(warped_ones)

    return compensated, mask

def compute_mask_and_preprocess_cpu(target_frame, blur_frames, imgsz, stride, ones_mask, pts_prev):
    """
    Preprocesses frames as per inference_jetson.py
    """
    blur1, blur_target, blur2 = blur_frames

    # compute motion compensation and differences
    comp1, _ = motion_compensate_cpu(blur1, blur_target, ones_mask, pts_prev)
    comp2, _ = motion_compensate_cpu(blur2, blur_target, ones_mask, pts_prev)
    
    frameDiff1 = cv2.absdiff(blur_target, comp1)
    frameDiff2 = cv2.absdiff(blur_target, comp2)

    # average the differences to create the continuous motion mask
    frameDiff = cv2.addWeighted(frameDiff1, 0.5, frameDiff2, 0.5, 0)

    # inflate 2D motion mask to 3D tensor
    frameDiff_3d = cv2.cvtColor(frameDiff, cv2.COLOR_GRAY2BGR)

    # pad and resize the image while maintaining aspect ratio
    target_img = letterbox_cpu(target_frame, new_shape = (imgsz, imgsz), color = (114, 114, 114), stride = stride, auto=False)
    mask_img = letterbox_cpu(frameDiff_3d, new_shape = (imgsz, imgsz), color = (0, 0, 0), stride = stride, auto=False)

    # convert BGR to RGB
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    
    return target_img, mask_img

def get_output_dirs(imgsz):
    """
    dynamically specify output folders depending on input image size
    """
    rgb_dir = PROJECT_ROOT / f"data/ptq_calibration_RGB_tensors_{imgsz}"
    mask_dir = PROJECT_ROOT / f"data/ptq_calibration_MASKS_tensors_{imgsz}"
    return rgb_dir, mask_dir

def extract_tensors(imgsz, frames_by_video):
    """
    overall function for handling data preprocessing - similar to preprocessing_worker() in
    inference_jetson.py
    """
    output_rgb_dir, output_mask_dir = get_output_dirs(imgsz)
    output_rgb_dir.mkdir(parents = True, exist_ok = True)
    output_mask_dir.mkdir(parents = True, exist_ok = True)

    stride = 32

    ones_mask = None
    pts_prev = None

    total_extracted = 0
    total_target = sum(len(frames) for frames in frames_by_video.values())

    print("-----------------------------------------------")
    print("Starting tensor extraction...")

    for vid_idx, (video_name, target_frames_dict) in enumerate(frames_by_video.items(), 1):
        video_path = TRAIN_VIDEOS_DIR / f"{video_name}.mp4"
        if not video_path.exists():
            print(f"Warning: Could not find {video_path}. Skipping.")
            continue
        
        target_frames = list(target_frames_dict.keys())
                
        # format target frames with their class for the print statement
        formatted_targets = [f"{f} ({target_frames_dict[f]})" for f in sorted(target_frames)]
        
        print("-----------------------------------------------")
        print(f"[{vid_idx}/{len(frames_by_video)}] Opening {video_name}.mp4")
        print(f"Searching for {len(target_frames)} frames: {', '.join(formatted_targets)}")
            
        cap = cv2.VideoCapture(str(video_path))
        
        # initialise buffers and state variables
        from collections import deque
        buffer = deque(maxlen = 5)
        frame_count = 0

        # reset homography matrices for each new video
        ones_mask = None
        pts_prev = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            frame_count += 1

            # print a trace every 100 frames
            if frame_count % 100 == 0:
                print(f"... scanning frame {frame_count} ...", flush = True)
            w0, h0 = frame.shape[1], frame.shape[0]

            # initialise homography mask per video
            if ones_mask is None:
                ones_mask = np.full((h0, w0), 255, dtype = np.uint8)

                # vectorised optical flow grid initialisation
                scale = 2
                gridSizeW, gridSizeH = 32 * scale, 24 * scale
                grid_numW = int((960 * scale) / gridSizeW - 1)
                grid_numH = int((540 * scale) / gridSizeH - 1)
                x_coords = np.arange(grid_numW, dtype=np.float32) * gridSizeW + gridSizeW / 2.0
                y_coords = np.arange(grid_numH, dtype=np.float32) * gridSizeH + gridSizeH / 2.0
                xv, yv = np.meshgrid(x_coords, y_coords, indexing = "ij")
                
                # convert coordinates into the 3D (batch, num_points, coordinates) matrix required for Lucas-Kanade 
                pts_prev = np.stack((xv.ravel(), yv.ravel()), axis=-1).reshape(-1, 1, 2).astype(np.float32)

            # apply greyscaling and blur to the original RGB frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (11, 11), 0)
            
            buffer.append((frame, blur))
            
            # wait for buffer to fill to five frames
            if len(buffer) < 5:
                continue
                
            # extract target frame data
            target_idx = frame_count - 2

            if target_idx in target_frames:
                target_frame = buffer[2][0]
                blur_frames = (buffer[0][1], buffer[2][1], buffer[4][1])

                target_img, mask_img = compute_mask_and_preprocess_cpu(
                    target_frame,
                    blur_frames,
                    imgsz,
                    stride,
                    ones_mask,
                    pts_prev
                )

                # convert to PyTorch tensors to mimic inference_jetson.py
                # shape becomes (1, 3, 1280, 1280)
                img1_torch = torch.from_numpy(target_img).permute(2, 0, 1).unsqueeze(0)
                img2_torch = torch.from_numpy(mask_img).permute(2, 0, 1).unsqueeze(0)

                img1 = img1_torch.contiguous().float() / 255.0
                img2 = img2_torch.contiguous().float() / 255.0

                stem = f"{video_name}_{target_idx:04d}"
                np.save(output_rgb_dir / f"{stem}_rgb.npy", img1.numpy())
                np.save(output_mask_dir / f"{stem}_mask.npy", img2.numpy())

                total_extracted += 1
                frame_class = target_frames_dict[target_idx]
                print(f"Successfully saved frame {target_idx:04d} [{frame_class}] (Progress: {total_extracted}/{total_target})", flush = True)
                

            # early stopping loop
            if target_idx >= max(target_frames):
                print(f"Reached final target frame ({max(target_frames)}). Closing video early.", flush=True)
                print("-----------------------------------------------")
                break
                
        cap.release()
        print(f"[{vid_idx}/{len(frames_by_video)}] Extracted required tensors from {video_name}.")

    print("-----------------------------------------------")
    print(f"Extraction completed - extracted {total_extracted} .npy tensor pairs.")

if __name__ == "__main__":
    # sample the frames globally so the same frames are targeted for extraction and processing
    frames_by_video = sample_and_group_frames()

    # extract frames at the desired image sizes to match the models
    for imgsz in (640, 1280):
        print("-----------------------------------------------")
        print(f"=== Extracting {imgsz}px tensors ===")
        extract_tensors(imgsz, frames_by_video)