"""
this script takes trained YOLOMG .pt weights and uses them to 
detect drones in video input. It measures the performance of the detection
using mAP and FPS.
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
from data_prep_scripts.MOD_Functions import motion_compensate
from utils.augmentations import letterbox
from utils.metrics import box_iou
from utils.general import check_img_size
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
    
    # Add batch dimension (1, C, H, W)
    img = img.unsqueeze(0)
    
    return img, (h0, w0)

def process_batch(detections, labels, iouv):
    """
    Extracted from val.py: Return correct predictions matrix.
    Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
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

if __name__ == "__main__":
    pass