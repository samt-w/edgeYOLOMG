## This script converts .pt weights to ONNX

import torch
import torch.nn as nn
from models.experimental import attempt_load
from models.common import Conv
from models.yolo import Detect
from utils.activations import SiLU

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# path to the trained weights
weights = 'C:/Users/samta/programming/python/yolomg-copy/runs/train/ARD100_mask32-1280_uavs/weights/best.pt'  

model = attempt_load(weights, map_location=device)
model.eval()

# Swap to export-friendly activations and Detect settings so the traced
# ONNX graph is compact rather than expanding every SiLU into Sigmoid+Mul
# and every Detect grid-construction op into its full primitive scaffolding.
for k, m in model.named_modules():
    if isinstance(m, Conv) and isinstance(m.act, nn.SiLU):
        m.act = SiLU()
    elif isinstance(m, Detect):
        m.inplace = False
        m.onnx_dynamic = False

imgsz = 1280  # match whatever the model was originally trained at
im1 = torch.zeros(1, 3, imgsz, imgsz).to(device)  # x1: motion mask
im2 = torch.zeros(1, 3, imgsz, imgsz).to(device)  # x2: RGB frame

# Dry run, same as the stock export.py does, so the Detect layer's grid
# is built once under these settings before tracing for export.
with torch.no_grad():
    for _ in range(2):
        y = model(im1, im2)

torch.onnx.export(
    model, (im1, im2), 'yolomg_preview.onnx',
    input_names=['mask_input', 'rgb_input'],
    output_names=['output'],
    opset_version=12,
)
print('exported yolomg_preview.onnx')