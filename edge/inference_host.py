"""
this script is adapted from val.py. it is designed to run on a host device with
separate CPU/GPU memory

it takes trained YOLOMG .pt weights and uses them to detect drones 
in video input. It measures the performance of the detection using mAP and FPS.
"""

import sys
from config import TEST_VIDEOS_DIR, TEST_VIDEOS_DIR_MINIMUM, LABELS_TEST_DIR, PROJECT_ROOT
from pathlib import Path

# add repo root filepath dynamically to import modules from other directories
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2
import torch
import numpy as np
from clearml import Task
from collections import deque
import time
import csv
from datetime import datetime
from queue import Queue
from threading import Thread

from data_prep_scripts.MOD_Functions import motion_compensate
from utils.augmentations import letterbox
from utils.metrics import box_iou, ap_per_class
from utils.general import check_img_size, non_max_suppression, xywhn2xyxy, scale_coords
from utils.torch_utils import select_device
from models.experimental import attempt_load

from concurrent.futures import ThreadPoolExecutor

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
    # compute motion compensation and differences
    img_compensate1, _, _, _, _, _ = motion_compensate(lastFrame1, lastFrame2)
    img_compensate2, _, _, _, _, _ = motion_compensate(currentFrame, lastFrame2)
    frameDiff1 = cv2.absdiff(lastFrame2, img_compensate1)    
    frameDiff2 = cv2.absdiff(lastFrame2, img_compensate2)

    # average the differences to create the continuous motion mask
    frameDiff = cv2.addWeighted(frameDiff1, 0.5, frameDiff2, 0.5, 0) 
    
    # inflate 2D motion mask to 3D tensor
    frameDiff_3d = cv2.cvtColor(frameDiff, cv2.COLOR_GRAY2BGR)
    return frameDiff_3d

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
    
    model = attempt_load(weights, map_location = device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s = stride) # ensure image size is a multiple of the max stride
    
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
                     pad_colour):
    """
    Resizes, pads, and normalises the numpy image array into a CPU tensor
    
    Args:
        image: the raw BGR image array 
        imgsz: target image inference size
        stride: model's maximum stride (for padding calculation)
        pad_colour: a three-integer tuple setting the colour of the letterbox padding (should be black if no motion)
        
    Returns a tuple (tensor_image, (h0, w0)) where tensor_image is the processed tensor 
    and (h0, w0) is the original image height and width
    """
    h0, w0 = image.shape[:2]
    
    # pad and resize the image while maintaining aspect ratio
    img = letterbox(im = image,
                    new_shape = (imgsz, imgsz),
                    color = pad_colour, # force specific padding colour
                    auto = True, # force minimum-stride padding to match val.py's "rect = False if task == 'benchmark' else pt"
                    stride = stride)[0]
    
    # convert from OpenCV's default BGR to RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)  # optimise memory layout for pytorch
    
    # convert to CPU tensor
    img = torch.from_numpy(img)
    
    # add batch dimension to fit pytorch batch logic and pin memory for fast GPU transfer
    # adds a dummy dimension at index 0 to create a 4D tensor of 1 image in the batch (1, C, H, W)
    img = img.unsqueeze(0).pin_memory()
    
    return img, (h0, w0)

def execute_prediction(model,
                  img1,
                  img2,
                  device,
                  conf_thres = 0.001,
                  iou_thres = 0.4):
    """
    Executes forward pass on a single frame and apply non-maximum suppression
    
    Args:
        model: the loaded model
        img1: the RGB frame tensor
        img2: the motion mask tensor
        device: specify the device for timing synchronisation
        conf_thres: object confidence threshold (default is the standard mAP 0.001)
        iou_thres: IoU threshold (default is the standard NMS 0.4)
        
    Returns a tensor of detections containing [x1, y1, x2, y2, confidence score, class]
    and the time taken for the forward pass and NMS
    """
    if device.type != "cpu":
        # initialise timing events
        start_inf = torch.cuda.Event(enable_timing=True)
        end_inf = torch.cuda.Event(enable_timing=True)
        end_nms = torch.cuda.Event(enable_timing=True)
        # start timing inference
        start_inf.record()
        with torch.no_grad():
            # forward pass
            pred = model(img1, img2, augment = False)[0]

        # end timing inference (will also use this to start timing NMS)
        end_inf.record()
            
        # apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]

        # end timing NMS
        end_nms.record()

        # return unsynchronised timing events
        return pred, start_inf, end_inf, end_nms

    else:
        # logic for executing and timing CPU runs
        t0 = time.time()
        with torch.no_grad():
            pred = model(img1, img2, augment = False)[0]
        t1 = time.time()
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]
        t2 = time.time()
        return pred, (t1 - t0), (t2 - t1)

def load_labels(label_path,
                h0,
                w0,
                min_area = 25):
    """
    Read YOLO-format ground truth labels from a .txt file and convert to absolute coordinates
    
    Args:
        label_path: path to the .txt file
        h0: original image height
        w0: original image width
        min_area: minimum area (in pixels) for a bounding box to be valid (original repo set 25px)
        
    Returns a tuple (labels, true_class_indices) where labels (num_labels, 5) contains
    absolute coordinates [class, x1, y1, x2, y2], and true_class_indices is a list
    """
    true_class_indices = []
    # initialise the empty labels array
    labels_pt_tensor = torch.zeros((0, 5))
    
    if label_path.exists():
        with open(label_path, "r") as f:
            # read non-empty lines
            labels = [x.split() for x in f.read().strip().splitlines() if len(x)]
            if len(labels):
                labels = np.array(labels, dtype = np.float32)
                
                # convert normalised [x_center, y_center, width, height] to absolute [x1, y1, x2, y2]
                labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], w = w0, h = h0)

                # apply area filter (same logic as original repo)
                areas = (labels[:, 3] - labels[:, 1]) * (labels[:, 4] - labels[:, 2])
                valid_mask = areas >= min_area
                labels = labels[valid_mask]
                
                # only include frames with positive class labels
                if len(labels):
                    true_class_indices = labels[:, 0].tolist()
                    # create labels tensor and pin memory for fast GPU transfer 
                    labels_pt_tensor = torch.from_numpy(labels).pin_memory()
                
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

def calculate_metrics(timings,
                      stats,
                      names,
                      inference_count,
                      run_config):
    """
    Calculates, prints, and saves the final FPS and mAP metrics
    
    Args:
        timings: dictionary containing accumulated frame times for each pipeline step
        stats: statistics tuple from evaluate_frame()
        names: dictionary mapping class indices to string names
        inference_count: total number of valid frames processed
        run_config: dictionary with configuration data for saving to CSV
    """
    print("\n" + "="*40)
    print(f"RESULTS FOR VIDEO {run_config['video_name']}")
    print("="*40)

    metrics = {
        "FPS_Full_Pipeline": 0.0,
        "FPS_Inference_Only": 0.0,
        "Read_ms": 0.0,
        "Mask_ms": 0.0,
        "Prep_ms": 0.0,
        "Inf_ms": 0.0,
        "NMS_ms": 0.0,
        "Total_ms": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "mAP_50": 0.0,
        "mAP_50_95": 0.0
    }
    
    # latency/FPS calculation
    if len(timings["total"]) > 0:
        mean_total = np.mean(timings["total"])
        mean_inf = np.mean(timings["inf"])
        mean_nms = np.mean(timings["nms"])
        metrics["FPS_Full_Pipeline"] = 1.0 / mean_total
        metrics["FPS_Inference_Only"] = 1.0 / (mean_inf + mean_nms) if (mean_inf + mean_nms) > 0 else 0.0
        metrics["Read_ms"] = np.mean(timings["read"]) * 1000
        metrics["Mask_ms"] = np.mean(timings["mask"]) * 1000
        metrics["Prep_ms"] = np.mean(timings["prep"]) * 1000
        metrics["Inf_ms"] = mean_inf * 1000
        metrics["NMS_ms"] = mean_nms * 1000
        metrics["Total_ms"] = mean_total * 1000

        print(f"Total number of Processed Frames: {inference_count}")
        print(f"Effective Real-Time FPS: {metrics['FPS_Full_Pipeline']:.2f} FPS")
        print(f"Inference-Only FPS:      {metrics['FPS_Inference_Only']:.2f} FPS")
        print("\nPipeline Step Latencies (mean):")
        print(f"Frame Read:  {metrics['Read_ms']:.2f} ms")
        print(f"Motion Mask: {metrics['Mask_ms']:.2f} ms")
        print(f"Preprocess:  {metrics['Prep_ms']:.2f} ms")
        print(f"Inference:   {metrics['Inf_ms']:.2f} ms")
        print(f"NMS:         {metrics['NMS_ms']:.2f} ms")
        print(f"Total time:  {metrics['Total_ms']:.2f} ms")
    else:
        print("Not enough frames processed to calculate FPS.")

    # accuracy/mAP calculation
    if not stats:
        print("\nNo targets or predictions found. Cannot calculate mAP.")
    else:
        # convert stats tuples into arrays
        stats_combined = [np.concatenate(x, 0) for x in zip(*stats)]
        
        if len(stats_combined) and stats_combined[0].any():
            # compute precision, recall, and AP per class
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats_combined,
                                                          plot = False,
                                                          save_dir = Path(""),
                                                          names = names)
            
            # AP at IoU 0.5
            ap50 = ap[:, 0]
            # AP averaged across IoU 0.5 to 0.95
            ap_095 = ap.mean(1)
            
            # compute mean over all classes
            metrics["Precision"] = p.mean()
            metrics["Recall"] = r.mean()
            metrics["mAP_50"] = ap50.mean()
            metrics["mAP_50_95"] = ap_095.mean()
            
            print("\nAccuracy Metrics:")
            print(f"Precision:    {metrics['Precision']:.4f}")
            print(f"Recall:       {metrics['Recall']:.4f}")
            print(f"mAP@0.5:      {metrics['mAP_50']:.4f}")
            print(f"mAP@0.5:0.95: {metrics['mAP_50_95']:.4f}")
        else:
            print("\nNo targets or predictions found. Cannot calculate mAP.")

    print("="*40)

    # Save results to CSV
    output_dir = run_config["output_dir"]
    csv_file = output_dir / f"{run_config['run_group_id']}_inference-results.csv"
    file_exists = csv_file.is_file()
    
    headers = [
        "Run_Group_ID", "Video", "Resolution", "Conf_Thres", "IoU_Thres", "Weights",
        "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", 
        "FPS(Pipeline)", "FPS(Inference_Only)", 
        "Read(ms)", "Mask(ms)", "Prep(ms)", "Inference(ms)", "NMS(ms)", "Total(ms)"
    ]
    
    row_data = [
        run_config["run_group_id"],
        run_config["video_name"],
        run_config["imgsz"],
        run_config["conf_thres"],
        run_config["iou_thres"],
        run_config["weights"],
        f"{metrics['Precision']:.4f}",
        f"{metrics['Recall']:.4f}",
        f"{metrics['mAP_50']:.4f}",
        f"{metrics['mAP_50_95']:.4f}",
        f"{metrics['FPS_Full_Pipeline']:.2f}",
        f"{metrics['FPS_Inference_Only']:.2f}",
        f"{metrics['Read_ms']:.2f}",
        f"{metrics['Mask_ms']:.2f}",
        f"{metrics['Prep_ms']:.2f}",
        f"{metrics['Inf_ms']:.2f}",
        f"{metrics['NMS_ms']:.2f}",
        f"{metrics['Total_ms']:.2f}"
    ]
    
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row_data)
        
    print(f"Results appended to {csv_file.resolve()}")

# initialise threaded reader class to queue frames for GPU
class VideoReader:
    def __init__(self, video_filepath, queue_size = 30):
        self.videocap = cv2.VideoCapture(str(video_filepath))
        # initialise a queue capped at the maximum size
        self.framequeue = Queue(maxsize = queue_size)
        # flag to stop the thread at the end of the video
        self.stopped = False
        # initialise a thread running the update() function continuously
        Thread(target = self.update, daemon = True).start()

    def update(self):
        while not self.stopped:
            # read the next frame from the video, and time the operation
            t_start = time.time()
            ret, frame = self.videocap.read()
            t_end = time.time()

            if not ret:
                # terminate the looping thread
                self.stopped = True
                return

            # add the frame and the read time to the back of the queue
            t_read = t_end - t_start
            self.framequeue.put((frame, t_read))
                
    def read(self):
        # if video is finished and queue is empty, return False
        if self.stopped and self.framequeue.empty():
            return False, None, 0.0
        
        # else, return True, unpack the frame and read time, and remove the oldest frame (i.e. at the front of the queue)
        frame, t_read = self.framequeue.get()
        return True, frame, t_read

    def isOpened(self):
        # necessary check for the OpenCV object
        return self.videocap.isOpened()
        
    def release(self):
        # match the OpenCV logic by stopping the thread and releasing the file
        self.stopped = True
        self.videocap.release()

def process_single_frame_payload(target_frame,
                                 mask_frames,
                                 target_frame_idx,
                                 video_name,
                                 label_dir,
                                 imgsz,
                                 stride,
                                 target_t_read,
                                 max_workers):
    """
    Helper function to compute mask, preprocess images, and load labels for a single frame.
    Runs inside the multi-frame thread pool.

    Args:
        target_frame: raw array for current frame
        mask_frames: tuple of three preprocessed arrays (lastFrame1, targetFrame, currentFrame) to compute motion mask
        target_frame_idx: frame number for target_frame, used to locate correct label
        video_name: name of current video (used to locate labels)
        label_dir: directory path containing YOLO .txt labels
        imgsz: target image size
        stride: model's maximum stride, used for letterbox padding
        max_workers: the number of permitted threads
    """
    h0_orig, w0_orig = target_frame.shape[:2]
    label_path = label_dir / f"{video_name}_{target_frame_idx:04d}.txt"
    labels, true_class_indices = load_labels(label_path, h0_orig, w0_orig)
    
    if len(labels) == 0:
        return None  # skip empty frames
    
    t_mask_start = time.time()
    mask = compute_mask_in_memory(mask_frames[0], mask_frames[1], mask_frames[2])
    t_mask_end = time.time()
    
    t_prep_start = time.time()
    img1, (h0, w0) = preprocess_image(target_frame, imgsz, stride, pad_colour=(114, 114, 114))
    img2, _ = preprocess_image(mask, imgsz, stride, pad_colour=(0, 0, 0))
    t_prep_end = time.time()
    
    return {
        "img1": img1,
        "img2": img2,
        "labels": labels,
        "true_class_indices": true_class_indices,
        "h0": h0,
        "w0": w0,
        "t_read": target_t_read,
        "t_mask": (t_mask_end - t_mask_start) / max_workers, # time per finished frame
        "t_prep": (t_prep_end - t_prep_start) / max_workers # time per finished frame
    }

def data_prep_worker(video_name,
                     cap,
                     label_dir,
                     imgsz,
                     stride,
                     inference_queue,
                     max_workers = 3):
    """
    Background worker multithreads that handle CPU-bound data preprocessing

    Uses frame indexes to ensure frame ordering in inference queue is correct

    Args:
        video_name: name of current video (used to locate labels)
        cap: OpenCV object of the current video
        label_dir: directory path containing YOLO .txt labels
        imgsz: target image size
        stride: model's maximum stride, used for letterbox padding
        inference_queue: Queue object where processed data is placed for the GPU

    Sends dictionary of tensors and labels to the inference queue
    """
    frame_buffer = deque(maxlen = 5)
    frame_count = 0
    # starting indexing at 3 to match the target frame index
    next_output_idx = 3
    pending_futures = {}

    # threads process max 3 frames concurrently on the CPU
    with ThreadPoolExecutor(max_workers = max_workers) as pool:
        while cap.isOpened():
            ret, frame, t_read = cap.read()
            
            if not ret or frame is None:
                break
                
            frame_count += 1
            # greyscale and blur frame for feeding to motion mask function
            grey_blur_frame = cv2.cvtColor(cv2.GaussianBlur(frame, (11, 11), 0), cv2.COLOR_BGR2GRAY)
            # add frame, processed frame, and read time to buffer
            frame_buffer.append((frame, grey_blur_frame, t_read))

            # skip inference until 5-frame buffer populated
            if len(frame_buffer) < 5:
                continue

            target_frame = frame_buffer[2][0]
            target_t_read = frame_buffer[2][2]
            target_frame_idx = frame_count - 2

            mask_frames = (frame_buffer[0][1], frame_buffer[2][1], frame_buffer[4][1])

            # submit frames to pool
            future = pool.submit(process_single_frame_payload,
                                 target_frame,
                                 mask_frames,
                                 target_frame_idx,
                                 video_name,
                                 label_dir,
                                 imgsz,
                                 stride,
                                 target_t_read,
                                 max_workers)

            pending_futures[target_frame_idx] = future

            # REORDERING BUFFER
            # push completed frames to inference_queue in frame index order, NOT completion order
            
            # check whether the next frame has been read and submitted as a future            
            while next_output_idx in pending_futures:
                fut = pending_futures[next_output_idx]

                # check whether next frame has completed processing - if not, exit and try again in next loop
                if not fut.done():
                    break

                # if next frame has completed, advance the pointer
                payload = fut.result()
                del pending_futures[next_output_idx]
                next_output_idx += 1
        
                # push completed frame to queue (blocks by default if queue hits maxsize)
                if payload is not None:
                    inference_queue.put(payload)

        # remove remaining completed futures at end of frame
        while pending_futures:
            if next_output_idx in pending_futures:
                fut = pending_futures[next_output_idx]
                payload = fut.result()
                del pending_futures[next_output_idx]
                next_output_idx += 1
                if payload is not None:
                    inference_queue.put(payload)
            else:
                next_output_idx += 1
        
    # add None to queue at end of video to tell GPU the video is finished
    inference_queue.put(None)

def run_inference_directory(video_dir,
                            label_dir,
                            weights,
                            imgsz,
                            device_id,
                            conf_thres = 0.001,
                            iou_thres = 0.4,
                            warmup_frames = 30):
    """
    Collates the inference pipeline using multithreading:
    - video loading
    - frame buffering
    - pipeline timing
    - inference evaluation and metric reporting
    - ClearML task logging
    
    Args:
        video_dir: path to the directory of input videos
        label_dir: directory containing frame .txt labels
        weights: path to model .pt weights
        imgsz: image size
        device_id: hardware device ID (e.g. "0" for GPU)
        conf_thres: object confidence threshold (default is the standard mAP 0.001)
        iou_thres: IoU threshold (default is the standard NMS 0.4)
        warmup_frames: number of frames to ignore in latency calculations (default is 30)
    """
    video_dir = Path(video_dir)
    label_dir = Path(label_dir)

    # create run identifier
    run_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_group_id = f"YOLOMG-Inference-Run_{run_time}"

    # initialise ClearML task
    task = Task.init(
        project_name="YOLOMG-STW",
        task_name=f"YOLOMG-STW-inference_{run_time}",
        output_uri=False,
        reuse_last_task_id=False
    )

    # initialise model
    print(f"Loading weights from {weights}...")
    model, device, half, stride, imgsz, names = load_model_and_device(weights, device_id, imgsz)
    
    print("Warming up CUDA context...")
    warmup_model(model, device, imgsz, warmup_iterations = 3)
    
    # initialise array for mAP50:90 calculation
    iou_vector = torch.linspace(0.5, 0.95, 10, device = device)
    iou_num = iou_vector.numel()

    # initialise list of test videos
    video_files = list(video_dir.glob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in {video_dir}")
        task.close()
        return

    # initialise empty structures for overall performance summary
    overall_timings = {"total": [], "read": [], "mask": [], "prep": [], "inf": [], "nms": []}
    overall_stats = []
    overall_inference_count = 0

    global_inference_count = 0 # for enabling pipeline CUDA warm-up

    for video_path in video_files:
        video_name = video_path.stem
    
        # initialise video and inference queue
        cap = VideoReader(video_path)
        inference_queue = Queue(maxsize = 30)

        # initialise background thread for data preprocessing
        prep_thread = Thread(target = data_prep_worker,
                             args = (video_name,
                                     cap,
                                     label_dir,
                                     imgsz,
                                     stride,
                                     inference_queue),
                             daemon = True)
        prep_thread.start()

        video_inference_count = 0
        video_stats = []

        # create dictionaries to hold timings
        video_timings = {
            "total": [],
            "read": [],
            "mask": [],
            "prep": [],
            "inf": [],
            "nms": []
        }
        
        print(f"Starting inference evaluation for {video_name}...")
        
        # record start time of first frame
        last_frame_end_time = time.time() 
        
        while True:
            # pull next frame's pre-packaged data from the queue
            payload = inference_queue.get()
            
            # if next item is 'None', video is finished
            if payload is None:
                break
                
            # unpack data
            img1_cpu = payload["img1"]
            img2_cpu = payload["img2"]
            labels = payload["labels"]
            true_class_indices = payload["true_class_indices"]
            h0 = payload["h0"]
            w0 = payload["w0"]
            # run inference and non-maximum suppression
            if device.type != 'cpu':
                # transfer CPU tensor to GPU
                img1 = img1_cpu.to(device, non_blocking = True)
                img1 = img1.half() if half else img1.float()
                img1 /= 255.0
                
                img2 = img2_cpu.to(device, non_blocking = True)
                img2 = img2.half() if half else img2.float()
                img2 /= 255.0

                # transfer labels to GPU
                labels = labels.to(device, non_blocking = True)

                # execute prediction and record timings
                pred, inf_start_evt, inf_end_evt, nms_end_evt = execute_prediction(model, img1, img2, device, conf_thres, iou_thres)
                
                # synchronise the pipeline timings
                nms_end_evt.synchronize() 
                
                # compute GPU kernel times
                t_inf = inf_start_evt.elapsed_time(inf_end_evt) / 1000.0
                t_nms = inf_end_evt.elapsed_time(nms_end_evt) / 1000.0
            
            
            else:
                pred, t_inf, t_nms = execute_prediction(model, img1, img2, device, conf_thres, iou_thres)

            # compute pipeline throughput time for this frame
            current_time = time.time()
            t_total = current_time - last_frame_end_time
            last_frame_end_time = current_time

            video_inference_count += 1
            global_inference_count += 1
            
            # only record frame times after the warmup period, to avoid spoilt averages
            if video_inference_count > warmup_frames:
                video_timings["read"].append(payload["t_read"])
                video_timings["mask"].append(payload["t_mask"])
                video_timings["prep"].append(payload["t_prep"])
                video_timings["inf"].append(t_inf)
                video_timings["nms"].append(t_nms)
                video_timings["total"].append(t_total)

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
                video_stats.append(frame_stats)

            # running report on pipeline FPS
            if video_inference_count % 100 == 0:
                current_fps = 1.0 / np.mean(video_timings["total"]) if video_timings["total"] else 0.0
                print(f"Processed {video_inference_count} frames for {video_name}... Current Pipeline FPS: {current_fps:.2f}")

        cap.release()

        # define output directory
        output_dir = Path(__file__).resolve().parent / "inference_results"
        output_dir.mkdir(parents = True, exist_ok = True)

        # sanitise the weights path for git tracking/Github uploading
        try:
            # make the weights path relative to project root directory
            sanitised_weights = Path(weights).resolve().relative_to(ROOT)
        except ValueError:
            # in case weights are stored outside project root (but will just store the final filename)
            sanitised_weights = Path(weights).name

        # create config for CSV export for this video
        run_config = {
            "run_group_id": run_group_id,
            "video_name": video_name,
            "imgsz": imgsz,
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
            "weights": str(sanitised_weights),
            "output_dir": output_dir
        }

        # compute and display metrics
        names_dict = dict(enumerate(names))
        calculate_metrics(video_timings, video_stats, names_dict, video_inference_count, run_config)
        # add metrics to overall summary
        for key in overall_timings:
            overall_timings[key].extend(video_timings[key])
        overall_stats.extend(video_stats)
        overall_inference_count += video_inference_count

    # calculate metrics for the overall summary
    if overall_inference_count > 0:
        print("\n=== COMPUTING OVERALL TEST RUN SUMMARY ===")
        summary_config = run_config.copy()
        summary_config["video_name"] = "OVERALL_SUMMARY"
        calculate_metrics(overall_timings, overall_stats, names_dict, overall_inference_count, summary_config)

    # upload updated inference_results.csv to ClearML as an artifact
    output_dir = Path(__file__).resolve().parent / "inference_results"
    csv_path = output_dir / f"{run_group_id}_inference-results.csv"
    if csv_path.exists():
        task.upload_artifact(
            name = f"Inference_Results_{run_group_id}",
            artifact_object = str(csv_path)
        )
        print(f"Uploaded {csv_path} as artifact to ClearML task {task.id}")

    task.close()

if __name__ == "__main__":
    TEST_VIDEOS_DIR = TEST_VIDEOS_DIR
    LABEL_DIR = LABELS_TEST_DIR
    WEIGHTS_640 = PROJECT_ROOT / "weights/best_640.pt"
    WEIGHTS_1280 = PROJECT_ROOT / "weights/best_1280.pt"

    # FOR TESTING THIS INFERENCE PIPELINE WITH JUST A SINGLE VIDEO
    TEST_VIDEOS_DIR_MINIMUM = TEST_VIDEOS_DIR_MINIMUM

    run_inference_directory(video_dir = TEST_VIDEOS_DIR_MINIMUM,
                            label_dir = LABEL_DIR,
                            weights = WEIGHTS_1280,
                            imgsz = 1280,
                            device_id = "0",
                            conf_thres = 0.001,
                            iou_thres = 0.4,
                            warmup_frames = 30)