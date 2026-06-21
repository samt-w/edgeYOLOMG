import torch
from models.experimental import attempt_load

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weights = 'C:/Users/samta/programming/python/yolomg-copy/runs/train/ARD100_mask32-1280_uavs/weights/best.pt'  # path to your trained weights

model = attempt_load(weights, map_location=device)
model.eval()

imgsz = 1280  # match whatever you trained at
im1 = torch.zeros(1, 3, imgsz, imgsz).to(device)  # x1: motion mask
im2 = torch.zeros(1, 3, imgsz, imgsz).to(device)  # x2: RGB frame

torch.onnx.export(
    model, (im1, im2), 'yolomg_preview.onnx',
    input_names=['mask_input', 'rgb_input'],
    output_names=['output'],
    opset_version=12,
)
print('exported yolomg_preview.onnx')