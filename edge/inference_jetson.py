"""
this script is adapted from inference_host.py - it runs on an edge device (Jetson Orin Nano Super)
with unified CPU/GPU memory

it takes a TensorRT .engine file compiled from trained YOLOMG .pt weights and uses them to detect drones 
in video input. It measures the performance of the detection using mAP and FPS.
"""

import sys
from pathlib import Path

# add repo root filepath dynamically to import modules from other directories
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import TEST_VIDEOS_DIR, TEST_VIDEOS_DIR_MINIMUM, LABELS_TEST_DIR, PROJECT_ROOT

import cv2
import torch
import numpy as np
import cupy as cp
from clearml import Task
from collections import deque
import time
import csv
from datetime import datetime
from queue import Queue
from threading import Thread

from data_prep_scripts.MOD_Functions import motion_compensate_cuda
from utils.augmentations import letterbox_cuda
from utils.metrics import box_iou, ap_per_class
from utils.general import check_img_size, non_max_suppression, xywhn2xyxy, scale_coords
from utils.torch_utils import select_device
from models.common import DetectMultiBackend

def compute_mask_and_preprocess_cuda(target_frame_gpu,
                                     blur_frames_gpu,
                                     imgsz,
                                     stride,
                                     lk_solver,
                                     original_dims,
                                     ones_gpu,
                                     pts_prev_cpu,
                                     pts_prev_gpu,
                                     cv_stream):
    """
    This function computes ego motion compensation (using Lucas-Kanade), a motion 
    mask, and letterboxing for three sequential frames (which have previously
    been resized, blurred and grescaled)

    Args:
        target_frame_gpu: the raw BGR current frame as a cv2.cuda_GpuMat
        blur_frames_gpu: a tuple of three pre-blurred cv2.cuda_GpuMat frames (lastFrame1, targetFrame, currentFrame)
        imgsz: target image inference size
        stride: model's maximum stride (for padding calculation)
        lk_solver: the global cv2.cuda.SparsePyrLKOpticalFlow object
        original_dims: the original dimensions of the image
        ones_gpu: a uniform matrix as a GpuMat object used for homography
        pts_prev_cpu: the global motion flow filter (on CPU)
        pts_prev_gpu: the global motion flow filter (on GPU)
        cv_stream: the CUDA stream for OpenCV
        
    Returns a tuple (img1, img2, (h0, w0), t_mask, t_prep) where img1 is the processed target frame 
    tensor as a GPU object, img2 is the motion mask tensor (also on GPU), (h0, w0) is the original image height and 
    width, and t_mask and t_prep are the measured processing times in seconds
    """
    t_mask_start = time.time()
    
    blur1_gpu, blur_target_gpu, blur2_gpu = blur_frames_gpu

    # compute motion compensation and differences
    comp1_gpu, _, _, _, _, _ = motion_compensate_cuda(blur1_gpu, blur_target_gpu, lk_solver, ones_gpu, pts_prev_cpu, pts_prev_gpu, cv_stream)
    comp2_gpu, _, _, _, _, _ = motion_compensate_cuda(blur2_gpu, blur_target_gpu, lk_solver, ones_gpu, pts_prev_cpu, pts_prev_gpu, cv_stream)
    frameDiff1 = cv2.cuda.absdiff(blur_target_gpu, comp1_gpu, stream = cv_stream)
    frameDiff2 = cv2.cuda.absdiff(blur_target_gpu, comp2_gpu, stream = cv_stream)

    # average the differences to create the continuous motion mask
    frameDiff = cv2.cuda.addWeighted(frameDiff1, 0.5, frameDiff2, 0.5, 0, stream = cv_stream)

    # inflate 2D motion mask to 3D tensor
    frameDiff_3d = cv2.cuda.cvtColor(frameDiff, cv2.COLOR_GRAY2BGR, stream = cv_stream)

    t_mask = time.time() - t_mask_start
    t_prep_start = time.time()

    # pad and resize the image while maintaining aspect ratio
    target_img_gpu = letterbox_cuda(target_frame_gpu, cv_stream, new_shape = (imgsz, imgsz), color = (114, 114, 114), stride = stride, auto=False)[0]
    mask_img_gpu = letterbox_cuda(frameDiff_3d, cv_stream, new_shape = (imgsz, imgsz), color = (0, 0, 0), stride = stride, auto=False)[0]

    # convert BGR to RGB on the GPU before DLPack bridge
    target_img_gpu = cv2.cuda.cvtColor(target_img_gpu, cv2.COLOR_BGR2RGB, stream = cv_stream)
    mask_img_gpu = cv2.cuda.cvtColor(mask_img_gpu, cv2.COLOR_BGR2RGB, stream = cv_stream)
    
    # retrieve dimensions for passing onwards
    h0, w0 = original_dims

    t_prep = time.time() - t_prep_start

    return target_img_gpu, mask_img_gpu, (h0, w0), t_mask, t_prep

def load_model_and_device(weights, device_id, imgsz, data_yaml):
    """
    Initialises the device and loads model weights
    
    Args:
        weights: path to the trained model weights (.pt file)
        device_id: device to use for inference (e.g., "0" for CUDA GPU 0)
        imgsz: inference image size
        data_yaml: the .yaml dataset file used to train the model (required for class labels)
        
    Returns a tuple with the loaded model, device object, boolean for FP16 precision,
    model stride, verified image size, and list of class names.
    """
    device = select_device(device_id)
    half = device.type != "cpu" # use half precision (FP16) only if using a GPU
    
    model = DetectMultiBackend(weights, device = device, fp16 = half, data = data_yaml)
    
    stride = int(model.stride)
    imgsz = check_img_size(imgsz, s = stride) # ensure image size is a multiple of the max stride
    
    model.eval() # set model to evaluation mode
    
    return model, device, half, stride, imgsz, model.names

def warmup_model(model,
                 device,
                 half,
                 imgsz,
                 batch_size,
                 warmup_iterations):
    """
    Run dummy forward passes to initialise CUDA context on GPUs
    
    Args:
        model: the loaded YOLO model
        device: the execution device
        half: specifying whether the data should be FP16
        imgsz: the image size used for inference
        batch_size: the number of images in each inference batch (must match TensorRT .engine)
        warmup_iterations: number of dummy passes to execute
    """
    if device.type != "cpu":
        # create a dummy tensor
        dummy_img = torch.zeros(batch_size, 3, imgsz, imgsz, device = device)
        dummy_img = dummy_img.half() if half else dummy_img.float()
        for _ in range(warmup_iterations):
            model(dummy_img, dummy_img)

def execute_forward_pass(model,
                         img1,
                         img2,
                         device):
    """
    Executes forward pass on a single frame but does not apply non-maximum suppression
    
    Args:
        model: the loaded model
        img1: the RGB frame tensor
        img2: the motion mask tensor
        device: specify the device for timing synchronisation
        
    Returns a tensor of raw unsuppressed predictions, and the start and end 
    timing events for the forward pass execution
    """
    if device.type != "cpu":
        # initialise timing events
        start_inf = torch.cuda.Event(enable_timing=True)
        end_inf = torch.cuda.Event(enable_timing=True)

        # start timing inference
        start_inf.record()
        with torch.no_grad():
            # forward pass
            pred = model(img1, img2, augment = False)
            # check whether the output is bundled as a list or tuple, and pass just 
            # the prediction tensor onwards
            if isinstance(pred, (list, tuple)):
                pred = pred[0]

        # end timing inference
        end_inf.record()

        # return unsynchronised timing events
        return pred, start_inf, end_inf
        
    else:
        # logic for executing and timing CPU runs
        t0 = time.time()
        with torch.no_grad():
            pred = model(img1, img2, augment = False)
            # check whether the output is bundled as a list or tuple, and pass just 
            # the prediction tensor onwards
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
        return pred, t0, time.time()

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
                    # create labels tensor
                    labels_pt_tensor = torch.from_numpy(labels)
                
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
        "Run_Group_ID", "Device", "Video", "Num_Frames", "Resolution", "Conf_Thres", "IoU_Thres", "Weights",
        "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", 
        "FPS(Pipeline)", "FPS(Inference_Only)", 
        "Read(ms)", "Mask(ms)", "Prep(ms)", "Inference(ms)", "NMS(ms)", "Total(ms)"
    ]
    
    row_data = [
        run_config["run_group_id"],
        "edge",
        run_config["video_name"],
        inference_count,
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

    # append overall summary data at end of video to master .csv document
    if run_config["video_name"] == "OVERALL_SUMMARY":
        master_csv_file = output_dir.parent / "YOLOMG-Inference-Runs_optimisation-results.csv"
        master_exists = master_csv_file.is_file()
        
        with open(master_csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not master_exists:
                writer.writerow(headers)
            writer.writerow(row_data)
            
        print(f"Overall summary appended to {master_csv_file.resolve()}")

# initialise reader class to queue frames for GPU
class VideoReader:
    def __init__(self, video_filepath, queue_size = 30):
        # GStreamer requires an absolute file path for its URI
        video_filepath_abs = Path(video_filepath).resolve().as_posix()
        
        # GStreamer pipeline for Jetson hardware decoding:
        # nvvidconv: hardware-accelerated memory/format conversion to BGRx
        # videoconvert: CPU conversion to standard BGR for OpenCV compatibility
        gst_pipeline = (
            f"filesrc location={video_filepath_abs} ! "
            "qtdemux ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink sync=false"
        )

        # attempt to initialise hardware-accelerated reading
        self.videocap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        # fallback mechanism: check if GStreamer opened successfully
        if not self.videocap.isOpened():
            print("\n" + "!" * 80, file=sys.stderr)
            print("ERROR: Failed to initialize GStreamer/NVDEC hardware decoding pipeline.", file=sys.stderr)
            print("Falling back to standard OpenCV CPU VideoCapture.", file=sys.stderr)
            print("Ensure OpenCV is compiled with GStreamer support and NVDEC is available.", file=sys.stderr)
            print("!" * 80 + "\n", file=sys.stderr)
            
            # Fall back to original OpenCV behavior
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
                self.framequeue.put(None)
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

def gpumat_to_torch(gpu_mat):
    """
    helper function converting a cv2.cuda_GpuMat object to a PyTorch Tensor on the GPU via CuPy/DLPack without copying to the CPU
    """
    # get tensor dimensions
    cols, rows = gpu_mat.size()
    channels = gpu_mat.channels()

    # get memory step (bytes per row)
    # will be computed if unavailable in the python binding
    try:
        step = gpu_mat.step
    except AttributeError:
        step = cols * channels # assuming uint8 data (1 byte per element)
    
    # extract the raw CUDA memory pointer from OpenCV
    mem = cp.cuda.UnownedMemory(gpu_mat.cudaPtr(), step * rows, gpu_mat)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    
    # wrap the pointer in a CuPy array
    cp_arr = cp.ndarray(shape = (rows, cols, channels),
                        dtype = cp.uint8,
                        memptr = memptr, # the memory pointer argument
                        strides = (step, channels, 1)) # stride logic: (bytes_per_row, bytes_per_pixel, bytes_per_element)
    
    # export the CuPy array to PyTorch via DLPack
    return torch.from_dlpack(cp_arr.toDlpack())
        
def gpu_pipeline_worker(input_queue,
                        output_queue,
                        model,
                        imgsz,
                        stride,
                        half,
                        device,
                        batch_size):
    """
    Background worker thread that handles GPU-bound preprocessing and inference.
    Pulls raw frames from the input queue, applies CUDA optical flow and letterboxing,
    executes the model forward pass, and pushes raw predictions to the output queue.

    Args:
        input_queue: Queue object providing raw frames and read times from the camera
        output_queue: Queue object where processed tensors and events are placed for the CPU
        model: the loaded TensorRT model
        imgsz: target image inference size
        stride: model's maximum stride
        half: specifying whether the data should be FP16
        device: specify the device for PyTorch execution

    Function is non-fruitful, just runs continuously until a None object is received 
    from the input queue
    """
    try:
        # initialise CUDA OpenCV objects locally within the thread to avoid
        # context-sharing clashes between CUDA streams
        lk_solver = cv2.cuda.SparsePyrLKOpticalFlow.create(winSize = (15, 15), maxLevel = 3)
        gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (11, 11), 0)
        
        # initialise a CUDA stream for the background thread and OpenCV
        worker_stream = torch.cuda.Stream(device)
        cv_stream = cv2.cuda_Stream()
        
        # initialise buffers and state variables
        buffer = deque(maxlen = 5)
        t_read_buffer = deque(maxlen = 5)
        ones_gpu = None
        # initialise empty variable for tracking points
        pts_prev_cpu = None
        pts_prev_gpu = None
        frame_count = 0
        # batch frame accumulation buffers
        batch_img1 = []
        batch_img2 = []
        batch_inference_data = []
        
        while True:
            # pull next frame data from queue
            item = input_queue.get()
            
            # if next item is None, video is finished 
            # (and need to remove remaining frame, if it exists)
            if item is None:
                # check if a partial batch remains
                if len(batch_img1) > 0:
                    valid_items = len(batch_img1)
                    # pad missing frame slot by duplicating the last frame
                    while len(batch_img1) < batch_size:
                        batch_img1.append(batch_img1[-1])
                        batch_img2.append(batch_img2[-1])
                    
                    with torch.cuda.stream(worker_stream):
                        batched_img1 = torch.cat(batch_img1, dim=0)
                        batched_img2 = torch.cat(batch_img2, dim=0)
                        pred, inf_start_evt, inf_end_evt = execute_forward_pass(model, batched_img1, batched_img2, device)
                    
                    # only push valid (non-duplicated) predictions to consumer
                    for i in range(valid_items):
                        meta = batch_inference_data[i]
                        payload = {
                            "pred": pred[i:i+1],
                            "inf_start_evt": inf_start_evt,
                            "inf_end_evt": inf_end_evt,
                            "batch_len": batch_size,
                            "frame_idx": meta["target_frame_idx"],
                            "w0": meta["w0"],
                            "h0": meta["h0"],
                            "t_read": meta["t_read"],
                            "t_mask": meta["t_mask"],
                            "t_prep": meta["t_prep"],
                            "img_shape": batched_img1.shape[2:]
                        }
                        output_queue.put(payload)
                
                output_queue.put(None)
                break
                
            # unpack data
            frame, t_read = item
            frame_count += 1
            
            # extract original image dimensions for downsampling
            w0, h0 = frame.shape[1], frame.shape[0]

            # initialise homography mask - doing it here so it is done just
            # once per video
            if ones_gpu is None:
                ones = np.full((h0, w0), 255, dtype = np.uint8)
                ones_gpu = cv2.cuda_GpuMat()
                ones_gpu.upload(ones, stream = cv_stream)

                # vectorised optical flow grid initialisation
                scale = 2
                width_grid = 960 * scale
                height_grid = 540 * scale
                gridSizeW = 32 * scale
                gridSizeH = 24 * scale
                
                grid_numW = int(width_grid / gridSizeW - 1)
                grid_numH = int(height_grid / gridSizeH - 1)
                
                # generate vectors with NumPy (replaces a previous Python for loop)
                x_coords = np.arange(grid_numW, dtype=np.float32) * gridSizeW + gridSizeW / 2.0
                y_coords = np.arange(grid_numH, dtype=np.float32) * gridSizeH + gridSizeH / 2.0
                
                # convert coordinates into the 3D (batch, num_points, coordinates) matrix required for Lucas-Kanade 
                xv, yv = np.meshgrid(x_coords, y_coords, indexing = "ij")
                pts_prev_cpu = np.stack((xv.ravel(), yv.ravel()), axis = -1).reshape(1, -1, 2)
                
                pts_prev_gpu = cv2.cuda_GpuMat()
                pts_prev_gpu.upload(pts_prev_cpu, stream = cv_stream)

            # convert each frame to GpuMat object
            frame_gpu = cv2.cuda_GpuMat()
            frame_gpu.upload(frame, stream = cv_stream)
            
            # apply greyscaling and blur to the original RGB frame            
            gray_gpu = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2GRAY, stream = cv_stream)
            blur_gpu = gaussian_filter.apply(gray_gpu, stream = cv_stream)
            
            # add full resolution GpuMat objects to the buffer
            buffer.append((frame_gpu, blur_gpu))
            t_read_buffer.append(t_read)
            
            # wait for buffer to fill to five frames
            if len(buffer) < 5:
                continue
                
            # extract target frame data
            target_frame_idx = frame_count - 2
            target_t_read = t_read_buffer[2]
            
            # preprocessing
            target_frame_gpu = buffer[2][0]
            blur_frames_gpu = (buffer[0][1], buffer[2][1], buffer[4][1])
            original_dims = (h0, w0)

            target_img_gpu, mask_img_gpu, _, t_mask, t_prep = compute_mask_and_preprocess_cuda(target_frame_gpu,
                                                                                               blur_frames_gpu,
                                                                                               imgsz,
                                                                                               stride,
                                                                                               lk_solver,
                                                                                               original_dims,
                                                                                               ones_gpu,
                                                                                               pts_prev_cpu,
                                                                                               pts_prev_gpu,
                                                                                               cv_stream)
            
            # wait for OpenCV to write the tensor
            cv_stream.waitForCompletion()
            
            # enable PyTorch to use the worker stream for memory transfer
            with torch.cuda.stream(worker_stream):
                # zero-copy convert GpuMat to PyTorch Tensor
                img1_torch = gpumat_to_torch(target_img_gpu)
                img2_torch = gpumat_to_torch(mask_img_gpu)
                
                # PyTorch channel slicing and transpositions
                # .permute(2, 0, 1) switches HWC to CHW
                # .unsqueeze(0) adds the batch dimension (1, C, H, W)
                img1_torch = img1_torch.permute(2, 0, 1).unsqueeze(0)
                img2_torch = img2_torch.permute(2, 0, 1).unsqueeze(0)
                
                # enable contiguous memory, and execute precision adjustment and pixel normalisation
                img1 = img1_torch.contiguous().half() if half else img1_torch.contiguous().float()
                img1 /= 255.0
                
                img2 = img2_torch.contiguous().half() if half else img2_torch.contiguous().float()
                img2 /= 255.0

            # add frame to batch
            batch_img1.append(img1)
            batch_img2.append(img2)
            batch_inference_data.append({
                "target_frame_idx": target_frame_idx,
                "w0": w0,
                "h0": h0,
                "t_read": target_t_read,
                "t_mask": t_mask,
                "t_prep": t_prep
            })

            # when batch capacity is reached, run batched forward pass
            if len(batch_img1) == batch_size:
                with torch.cuda.stream(worker_stream):
                    batched_img1 = torch.cat(batch_img1, dim = 0)
                    batched_img2 = torch.cat(batch_img2, dim = 0)
                    pred, inf_start_evt, inf_end_evt = execute_forward_pass(model, batched_img1, batched_img2, device)

                    # clone the predictions tensor so it does not get overwritten by the next batch
                    pred = pred.clone()
                
                # unpack batched predictions to individual queue items
                for i in range(batch_size):
                    meta = batch_inference_data[i]
                
                    # package raw tensors and events, push to evaluation queue
                    payload = {
                        "pred": pred[i:i+1],
                        "inf_start_evt": inf_start_evt,
                        "inf_end_evt": inf_end_evt,
                        "batch_len": batch_size,
                        "frame_idx": meta["target_frame_idx"],
                        "w0": meta["w0"],
                        "h0": meta["h0"],
                        "t_read": meta["t_read"],
                        "t_mask": meta["t_mask"],
                        "t_prep": meta["t_prep"],
                        "img_shape": batched_img1.shape[2:]
                    }
                    output_queue.put(payload)
                
                batch_img1.clear()
                batch_img2.clear()
                batch_inference_data.clear()

    except Exception:
        import traceback
        traceback.print_exc()
        output_queue.put(None)   # unblock the consumer instead of hanging it
        raise

def run_inference_directory(video_dir,
                            label_dir,
                            weights,
                            imgsz,
                            device_id,
                            data_yaml,
                            batch_size,
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
        data_yaml: the .yaml dataset file used to train the model (required for class labels)
        batch_size: the number of frames in each batch
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
    model, device, half, stride, imgsz, names = load_model_and_device(weights, device_id, imgsz, data_yaml)
    
    print("Warming up CUDA context...")
    warmup_model(model, device, half, imgsz, batch_size = batch_size, warmup_iterations = 3)
    
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

    for video_path in video_files:
        video_name = video_path.stem
        print(f"Starting asynchronous pipeline for {video_name}...")
        
        # initialise producer thread
        cap = VideoReader(video_path, queue_size = 30)
        eval_queue = Queue(maxsize = 30)
        
        # initialise GPU consumer thread
        gpu_thread = Thread(target = gpu_pipeline_worker,
                            args = (cap.framequeue,
                                    eval_queue,
                                    model,
                                    imgsz,
                                    stride,
                                    half,
                                    device,
                                    batch_size), daemon=True)
        gpu_thread.start()

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
                
        # record start time of first frame
        last_frame_end_time = time.time()

        # initialise CPU consumer thread
        while True:
            payload = eval_queue.get()

            # end processing if video is complete
            if payload is None:
                break

            # use .synchronize() to force CPU consumer thread to wait for GPU 
            # inference to finish before attempting to run NMS on the the tensor
            batch_len = payload.get("batch_len", 1)
            if device.type != "cpu":
                payload["inf_end_evt"].synchronize()
                # divide batch time by batch_len for per-frame time
                t_inf = (payload["inf_start_evt"].elapsed_time(payload["inf_end_evt"]) / 1000.0) / batch_len
            else:
                t_inf = (payload["inf_end_evt"] - payload["inf_start_evt"]) / batch_len


            # load Labels (extract original image dimensions from buffer)
            label_path = label_dir / f"{video_name}_{payload['frame_idx']:04d}.txt"
            labels, true_class_indices = load_labels(label_path, payload["h0"], payload["w0"])
            
            if len(labels) == 0:
                # loudly skip empty frames
                print(f"Warning: No labels found for frame {payload['frame_idx']:04d}. Skipping frame.")
                continue
            # transfer labels to GPU for IoU computation
            labels = labels.to(device)
            
            # conduct non-maximum suppression
            nms_start = time.time()

            # execute prediction and record timings
            pred_nms = non_max_suppression(payload["pred"], conf_thres, iou_thres)[0]            
            t_nms = time.time() - nms_start

            # compute pipeline throughput time for this frame
            current_time = time.time()
            t_total = current_time - last_frame_end_time
            last_frame_end_time = current_time

            video_inference_count += 1
            
            # only record frame times after the warmup period, to avoid spoilt averages
            if video_inference_count > warmup_frames:
                video_timings["read"].append(payload["t_read"])
                video_timings["mask"].append(payload["t_mask"])
                video_timings["prep"].append(payload["t_prep"])
                video_timings["inf"].append(t_inf)
                video_timings["nms"].append(t_nms)
                video_timings["total"].append(t_total)

            # evaluate predictions for this frame
            frame_stats = evaluate_frame(pred_nms,
                                         labels,
                                         true_class_indices,
                                         payload["h0"],
                                         payload["w0"],
                                         payload["img_shape"],
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
    # FOR TESTING THIS INFERENCE PIPELINE WITH JUST A SINGLE VIDEO
    TEST_VIDEOS_DIR_MINIMUM = TEST_VIDEOS_DIR_MINIMUM

    # dataset .yaml file (required for class labels for TensorRT engine)
    # if 640px and 1280px trained on same dataset, can just use one of their .yaml files
    DATA_YAML = PROJECT_ROOT / "data/ARD100_640.yaml"
    
    # # legacy .pt weights - not suitable for running on Jetson
    # PYTORCH_WEIGHTS_640 = PROJECT_ROOT / "weights/best_640.pt"
    # PYTORCH_WEIGHTS_1280 = PROJECT_ROOT / "weights/best_1280.pt"

    # TensorRT engine files compiled from .pt weights
    TENSORRT_WEIGHTS_640 = PROJECT_ROOT / "weights/best_640_jetpack_6-2-3.engine"
    TENSORRT_WEIGHTS_1280 = PROJECT_ROOT / "weights/best_1280_jetpack_6-2-3.engine"

    TENSORRT_WEIGHTS_640_MIXED_PRECISION = PROJECT_ROOT / "weights/best_640_jetpack_6-2-3_mixedprecision.engine"

    TENSORRT_WEIGHTS_640_BATCH2 = PROJECT_ROOT / "weights/best_640_jetpack_6-2-3_batch2.engine"
    TENSORRT_WEIGHTS_1280_BATCH2 = PROJECT_ROOT / "weights/best_1280_jetpack_6-2-3_batch2.engine"

    run_inference_directory(video_dir = TEST_VIDEOS_DIR_MINIMUM,
                            label_dir = LABEL_DIR,
                            weights = TENSORRT_WEIGHTS_640_BATCH2,
                            imgsz = 640, # MUST MATCH COMPILED WEIGHTS/ENGINE IMAGE SIZE
                            batch_size = 2, # MUST MATCH STATIC COMPILED ENGINE BATCH SIZE
                            device_id = "0",
                            data_yaml = DATA_YAML,
                            conf_thres = 0.001,
                            iou_thres = 0.4,
                            warmup_frames = 30)
    
    run_inference_directory(video_dir = TEST_VIDEOS_DIR_MINIMUM,
                            label_dir = LABEL_DIR,
                            weights = TENSORRT_WEIGHTS_1280_BATCH2,
                            imgsz = 1280, # MUST MATCH COMPILED WEIGHTS/ENGINE IMAGE SIZE
                            batch_size = 2, # MUST MATCH STATIC COMPILED ENGINE BATCH SIZE
                            device_id = "0",
                            data_yaml = DATA_YAML,
                            conf_thres = 0.001,
                            iou_thres = 0.4,
                            warmup_frames = 30)