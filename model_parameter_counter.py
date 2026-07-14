# this script counts the number of parameters in a .onnx model graph

import onnx
import numpy as np

model = onnx.load("yolomg_preview.onnx")

total = 0
breakdown = []

for init in model.graph.initializer:
    n = 1
    for d in init.dims:
        n *= d
    total += n
    breakdown.append((n, init.name, list(init.dims)))

breakdown.sort(reverse = True)

print(f"total parameters: {total:,}")
print(f"total initialiser tensors: {len(breakdown)}")