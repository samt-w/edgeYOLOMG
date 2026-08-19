import tensorrt as trt
import torch
import numpy as np
import os
import argparse
from pathlib import Path

# TensorRT requires a logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class DualInputCalibrator(trt.IInt8EntropyCalibrator2):
    """
    this is the class that feeds TensorRT data during INT8 calibration
    IInt8EntropyCalibrator2 uses KL divergence
    """
    def __init__(self, rgb_dir, mask_dir, batch_size, imgsz, cache_file):
        # initialise the parent C++ class 
        super(DualInputCalibrator, self).__init__()
        
        self.rgb_dir = Path(rgb_dir)
        self.mask_dir = Path(mask_dir)
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.cache_file = cache_file
        
        # gather all RGB .npy files
        self.rgb_files = sorted(list(self.rgb_dir.glob("*_rgb.npy")))
        if len(self.rgb_files) == 0:
            raise ValueError(f"No _rgb.npy files found in {self.rgb_dir}")

        # map RGB to Mask files to guarantee pairs exist 
        self.mask_files = []
        for rgb_path in self.rgb_files:
            expected_mask_name = rgb_path.name.replace("_rgb.npy", "_mask.npy")
            mask_path = self.mask_dir / expected_mask_name
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing corresponding mask file for {rgb_path.name}: Expected {mask_path}")
            self.mask_files.append(mask_path)
            
        self.num_images = len(self.rgb_files)
        # compute number of full batches and discard remainder images
        self.batches = self.num_images // self.batch_size
        self.current_batch = 0
        
        print(f"Found {self.num_images} exact RGB/Mask tensor pairs.")
        
        # pre-allocate pinned memory on the GPU via PyTorch, because
        # TensorRT calibrators require memory pointers, and initialising these
        # just once prevents fragmentation and OOM errors
        self.device_img = torch.zeros((self.batch_size, 3, self.imgsz, self.imgsz), dtype=torch.float32, device = "cuda")
        self.device_mask = torch.zeros((self.batch_size, 3, self.imgsz, self.imgsz), dtype=torch.float32, device = "cuda")

    def get_batch_size(self):
        # tell TensorRT how much data per batch
        return self.batch_size

    def get_batch(self, names):
        """
        this function will repeatedly give batches to TensorRT until 'None' is reached

        the 'names' argument is a list of strings representing input nodes from .onnx model
        """
        if self.current_batch >= self.batches:
            return None

        start_idx = self.current_batch * self.batch_size
        end_idx = start_idx + self.batch_size
        
        batch_imgs = []
        batch_masks = []
        
        for i in range(start_idx, end_idx):
            # load tensors and force FP32 as expected by TensorRT calibrator
            rgb_tensor = np.load(self.rgb_files[i]).astype(np.float32)
            mask_tensor = np.load(self.mask_files[i]).astype(np.float32)
            
            # remove batch dimension from saved tensor
            if rgb_tensor.ndim == 4 and rgb_tensor.shape[0] == 1:
                rgb_tensor = rgb_tensor[0]
            if mask_tensor.ndim == 4 and mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor[0]
                
            batch_imgs.append(rgb_tensor)
            batch_masks.append(mask_tensor)

        # convert NumPy arrays to PyTorch tensor, then transfer batch to GPU
        self.device_img.copy_(torch.tensor(np.array(batch_imgs)))
        self.device_mask.copy_(torch.tensor(np.array(batch_masks)))

        # ensure copy to GPU is complete before proceeding 
        torch.cuda.synchronize()
        
        self.current_batch += 1
        print(f"Calibrating batch {self.current_batch}/{self.batches}...")

        # map memory pointers to ONNX input names
        pointers = []
        for name in names:
            if name == 'images':
                pointers.append(int(self.device_img.data_ptr()))
            elif name == 'masks':
                pointers.append(int(self.device_mask.data_ptr()))
            else:
                raise ValueError(f"Unknown ONNX input name requested by calibrator: {name}")
                
        return pointers

    def read_calibration_cache(self):
        """
        called by TensorRT to recover batch cache file if it exists
        """
        if os.path.exists(self.cache_file):
            print(f"Loading existing calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """
        called by TensorRT to save cache once batch loop has finished
        """
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_int8_engine(onnx_path, engine_path, rgb_dir, mask_dir, imgsz, batch_size = 8):
    cache_file = f"yolomg_calib_{imgsz}.cache"
    calibrator = DualInputCalibrator(rgb_dir, mask_dir, batch_size, imgsz, cache_file)
    
    # initialise TensorRT builder
    builder = trt.Builder(TRT_LOGGER)
    
    # instruct TensorRT to read batch size from .onnx file
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # allocate workspace memory for TensorRT to use to test algorithms (here,
    # 4GB, represented as 4 * 2^30 bytes)
    if int(trt.__version__.split('.')[0]) >= 10:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1 << 30)
    else:
        config.max_workspace_size = 4 * 1 << 30

    # read the .onnx file and populate the network object that was initialised earlier
    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file.")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    # enable INT8 and explicitly enable FP16 fallback for unquantisable layers
    # note: without setting FP16 fallback, fallback will be FP32, which will damage performance
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    # attach the calibrator
    config.int8_calibrator = calibrator

    print(f"Building INT8 Engine for {imgsz}px... This will take 10-30 minutes.")
    
    # serialise network into a byte stream - split logic to reflect changes in TensorRT 10
    if int(trt.__version__.split('.')[0]) >= 10:
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            print("ERROR: Engine build failed. Check TensorRT logs.")
            return
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            print("ERROR: Engine build failed. Check TensorRT logs.")
            return
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
            
    print(f"Successfully saved INT8 engine to {engine_path}")

if __name__ == "__main__":
    # use argparse to ensure the script can be run in command line to create different input size model engines
    parser = argparse.ArgumentParser(description = "Compile YOLOMG to TensorRT INT8 Engine")
    parser.add_argument('--onnx', type = str, required = True, help = "Path to input .onnx model")
    parser.add_argument('--engine', type = str, required = True, help = "Path to output .engine model")
    parser.add_argument('--rgb_dir', type = str, required = True, help = "Path to calibration set RGB .npy directory")
    parser.add_argument('--mask_dir', type = str, required = True, help = "Path to calibration set Mask .npy directory")
    parser.add_argument('--imgsz', type = int, required = True, choices = [640, 1280], help="Image size (640 or 1280)")
    parser.add_argument('--batch', type = int, default = 8, help = "Calibration batch size")
    
    args = parser.parse_args()
    
    build_int8_engine(
        onnx_path = args.onnx, 
        engine_path = args.engine, 
        rgb_dir = args.rgb_dir, 
        mask_dir = args.mask_dir, 
        imgsz = args.imgsz,
        batch_size = args.batch
    )