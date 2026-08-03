"""
this script is adapted from val.py

it takes trained YOLOMG .pt weights and uses them to detect drones 
in video input. It measures the performance of the detection using mAP and FPS.
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
import torch
import numpy as np
from collections import deque
import time

from data_prep_scripts.MOD_Functions import motion_compensate
from utils.augmentations import letterbox
from utils.metrics import box_iou, ap_per_class
from utils.general import check_img_size, non_max_suppression, xywhn2xyxy, scale_coords
from utils.torch_utils import select_device
from models.experimental import attempt_load

def compute_mask_in_memory(lastFrame1, lastFrame2, currentFrame):
    """
    This function computes a motion mask for a particular frame, lastFrame2, given one preceding frame,
    lastFrame1, and and one succeeding frame, currentFrame.

    The temporal order of the frames is:
    - lastFrame1 (oldest)
    - lastFrame2
    - currentFrame (newest)

    This naming convention comes from the original code.

    This function differs to its data preparation counterpart because it holds the 
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

def load_model_and_device(weights, device_id, imgsz):
    """
    Initialises the device and loads model weights
    
    Args:
        weights: path to the trained model weights (.pt file)
        device_id: device to use for inference (e.g., "0" for CUDA GPU 0)
        imgsz: inference image size
        
    Returns a tuple with the loaded model, device object, boolean for FP16 precision,
    model stride, verified image size, and list of class names.
    """
    device = select_device(device_id)
    half = device.type != "cpu" # use half precision (FP16) only if using a GPU
    
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride) # ensure image size is a multiple of the max stride
    
    if half:
        model.half() # convert model weights to FP16 if using GPU
    model.eval() # set model to evaluation mode
    
    # get class names from model payload
    names = model.module.names if hasattr(model, 'module') else model.names
    
    return model, device, half, stride, imgsz, names

def warmup_model(model,
                 device, 
                 imgsz, 
                 warmup_iterations):
    """
    Run dummy forward passes to initialise CUDA context on GPUs
    
    Args:
        model: the loaded YOLO model
        device: the execution device
        imgsz: the image size used for inference
        warmup_iterations: number of dummy passes to execute
    """
    if device.type != "cpu":
        # create a dummy tensor
        dummy_img = torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        for _ in range(warmup_iterations):
            model(dummy_img, dummy_img)

def preprocess_image(image, 
                     imgsz,
                     stride,
                     device,
                     half):
    """
    Resizes, pads, and normalises the numpy image array into the pytorch tensor
    
    Args:
        image: the raw BGR image array 
        imgsz: target image inference size
        stride: model's maximum stride (for padding calculation)
        device: execution device
        half: sets whether to convert the tensor to FP16
        
    Returns a tuple (tensor_image, (h0, w0)) where tensor_image is the processed pytorch tensor 
    and (h0, w0) is the original image height and width
    """
    h0, w0 = image.shape[:2]
    
    # pad and resize the image while maintaining aspect ratio
    img = letterbox(im = image,
                    new_shape = (imgsz, imgsz),
                    stride = stride)[0]
    
    # convert from OpenCV's default BGR to RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)  # optimise memory layout for pytorch
    
    # convert to tensor and transfer to target device
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # convert uint8 to fp16/32
    img /= 255.0  # normalise pixel values from [0, 255] to [0, 1]
    
    # add batch dimension (1, C, H, W)
    img = img.unsqueeze(0)
    
    return img, (h0, w0)

def run_inference(model,
                  img1,
                  img2,
                  conf_thres = 0.001,
                  iou_thres = 0.4):
    """
    Executes forward pass on a single frame and apply non-maximum suppression
    
    Args:
        model: the loaded model
        img1: the RGB frame tensor
        img2: the motion mask tensor
        conf_thres: object confidence threshold (default is the standard mAP 0.001)
        iou_thres: IoU threshold (default is the standard NMS 0.4)
        
    Returns a tensor of detections containing [x1, y1, x2, y2, confidence score, class]
    """
    with torch.no_grad():
        # forward pass
        pred = model(img1, img2, augment = False)[0]
        
        # apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]
        
    return pred

def load_labels(label_path,
                      h0,
                      w0, 
                      device):
    """
    Read YOLO-format ground truth labels from a .txt file and convert to absolute coordinates
    
    Args:
        label_path: path to the .txt file
        h0: original image height
        w0: original image width
        device: device to place the label tensor on
        
    Returns a tuple (labels, true_class_indices) where labels (num_labels, 5) contains
    absolute coordinates [class, x1, y1, x2, y2], and true_class_indices is a list
    """
    true_class_indices = []
    # initialise the empty labels array
    labels_pt_tensor = torch.zeros((0, 5), device = device)
    
    if label_path.exists():
        with open(label_path, "r") as f:
            # read non-empty lines
            labels = [x.split() for x in f.read().strip().splitlines() if len(x)]
            if len(labels):
                labels = np.array(labels, dtype = np.float32)
                true_class_indices = labels[:, 0].tolist()
                
                # Convert normalised [x_center, y_center, width, height] to absolute [x1, y1, x2, y2]
                labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], w = w0, h = h0)
                labels_pt_tensor = torch.from_numpy(labels).to(device)
                
    return (labels_pt_tensor, true_class_indices)

def process_batch(detections, labels, iouv):
    """
    Extracted from val.py: Return correct predictions matrix.
    Both sets of boxes are in (x1, y1, x2, y2) format.
    Args:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
        iouv - the vector of IoU thresholds to be tested
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5])) # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy() # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def evaluate_frame(pred,
                   labels,
                   true_class_indices,
                   h0,
                   w0,
                   processed_shape,
                   iou_vector,
                   iou_num):
    """
    Evaluates predictions against ground truth labels for a single frame
    
    Args:
        pred: model predictions [N, 6]
        labels: ground truth labels [M, 5]
        true_class_indices: list of true class indices
        h0: original image height
        w0: original image width
        processed_shape: shape of the letterboxed image tensor (H, W)
        iou_vector: vector of IoU thresholds to test
        iou_num: total number of IoU thresholds
        
    Returns a tuple formatted for ap_per_class() - (correct, confidences, predicted_classes, true_classes)
    """
    if len(pred) == 0:
        if len(labels):
            return (torch.zeros(0, iou_num, dtype=torch.bool), torch.Tensor(), torch.Tensor(), true_class_indices)
        return None
        
    # rescale predictions from the padded image back to the native original image dimensions
    predn = pred.clone()
    predn[:, :4] = scale_coords(processed_shape, predn[:, :4], (h0, w0)).round()
    
    if len(labels):
        # compute IoU between predictions and true labels
        correct = process_batch(predn, labels, iou_vector)
    else:
        # if no positive class exists, all predictions are False
        correct = torch.zeros(pred.shape[0], iou_num, dtype = torch.bool)
        
    return (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), true_class_indices)

def calculate_metrics(frame_times,
                      stats,
                      names,
                      inference_count):
    """
    Calculates and prints the final FPS and mAP metrics
    
    Args:
        frame_times: list of time taken for each processed frame
        stats: statistics tuple from evaluate_frame()
        names: dictionary mapping class indices to string names
        inference_count: total number of valid frames processed
    """
    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print("="*40)
    
    # latency/FPS calculation
    if len(frame_times) > 0:
        mean_time = np.mean(frame_times)
        fps = 1.0 / mean_time
        print(f"Total number of Processed Frames: {inference_count}")
        print(f"Mean per-frame time: {mean_time*1000:.2f} ms")
        print(f"Effective Real-Time FPS: {fps:.2f} FPS")
    else:
        print("Not enough frames processed to calculate FPS.")

    # accuracy/mAP calculation
    if not stats:
        print("\nNo targets or predictions found. Cannot calculate mAP.")
        print("="*40)
        return

    # convert stats tuples into arrays
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    
    if len(stats) and stats[0].any():
        # compute precision, recall, and AP per class
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot = False,
                                                      save_dir = Path(''),
                                                      names = names)
        
        # AP at IoU 0.5
        ap50 = ap[:, 0]
        # AP averaged across IoU 0.5 to 0.95
        ap_095 = ap.mean(1)
        
        # compute mean over all classes
        mp, mr, map50, map_095 = p.mean(), r.mean(), ap50.mean(), ap_095.mean()
        
        print("\nAccuracy Metrics (Full Video):")
        print(f"Precision:    {mp:.4f}")
        print(f"Recall:       {mr:.4f}")
        print(f"mAP@0.5:      {map50:.4f}")
        print(f"mAP@0.5:0.95: {map_095:.4f}")
    else:
        print("\nNo targets or predictions found. Cannot calculate mAP.")
    print("="*40)

def run_inference(video_path,
                  label_dir,
                  weights,
                  imgsz,
                  device_id,
                  conf_thres = 0.001,
                  iou_thres = 0.4,
                  warmup_frames = 30):
    """
    Collates the inference pipeline:
    - video loading
    - frame buffering
    - pipeline timing
    - inference evaluation and metric reporting
    
    Args:
        video_path: path to the input video
        label_dir: directory containing frame .txt labels
        weights: path to model .pt weights
        imgsz: image size
        device_id: hardware device ID (e.g. "0" for GPU)
        conf_thres: object confidence threshold (default is the standard mAP 0.001)
        iou_thres: IoU threshold (default is the standard NMS 0.4)
        warmup_frames: number of frames to ignore in latency calculations (default is 30)
    """
    video_path = Path(video_path)
    label_dir = Path(label_dir)
    video_name = video_path.stem

    # initialise model
    print(f"Loading weights from {weights}...")
    model, device, half, stride, imgsz, names = load_model_and_device(weights, device_id, imgsz)
    
    print("Warming up CUDA context...")
    warmup_model(model, device, imgsz, warmup_iterations = 3)
    
    # initialise array for mAP50:90 calculation
    iou_vector = torch.linspace(0.5, 0.95, 10, device = device)
    iou_num = iou_vector.numel()

    # initialise video and buffer
    cap = cv2.VideoCapture(str(video_path))
    frame_buffer = deque(maxlen = 5)
    
    frame_count = 0
    inference_count = 0
    frame_times = []
    stats = []
    
    print(f"Starting inference evaluation for {video_name}...")
    
    while cap.isOpened():
        # start timing
        if device.type != 'cpu': 
            torch.cuda.synchronize(device)
        t_start = time.time()
        
        # read frame
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        frame_count += 1
        frame_buffer.append(frame)

        # skip inference until 5-frame buffer populated
        if len(frame_buffer) < 5:
            continue

        target_frame = frame_buffer[2]
        target_frame_idx = frame_count - 2
        
        # compute motion mask
        mask = compute_mask_in_memory(frame_buffer[0], frame_buffer[2], frame_buffer[4])
        
        # preprocess target images
        img1, (h0, w0) = preprocess_image(target_frame, imgsz, stride, device, half)
        img2, _ = preprocess_image(mask, imgsz, stride, device, half)

        # run inference and non-maximum suppression
        pred = run_inference(model, img1, img2, conf_thres, iou_thres)

        # end timing
        if device.type != 'cpu': 
            torch.cuda.synchronize(device)
        t_end = time.time()
        
        inference_count += 1
        
        # only record frame times after the warmup period, to avoid spoilt averages
        if inference_count > warmup_frames:
            frame_times.append(t_end - t_start)

        # load original labels
        label_path = label_dir / f"{video_name}_{target_frame_idx:04d}.txt"
        labels, true_class_indices = load_labels(label_path, h0, w0, device)

        # evaluate predictions
        frame_stats = evaluate_frame(pred,
                                     labels,
                                     true_class_indices,
                                     h0,
                                     w0,
                                     img1.shape[2:],
                                     iou_vector,
                                     iou_num)
        if frame_stats:
            stats.append(frame_stats)

        # running report on pipeline FPS
        if inference_count % 100 == 0:
            current_fps = 1.0 / np.mean(frame_times) if frame_times else 0.0
            print(f"Processed {inference_count} frames... Current Pipeline FPS: {current_fps:.2f}")

    cap.release()

    # compute and display metrics
    names_dict = dict(enumerate(names))
    calculate_metrics(frame_times, stats, names_dict, inference_count)

if __name__ == "__main__":
    pass