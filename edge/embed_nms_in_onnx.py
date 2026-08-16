"""
this is a custom script to amend YOLOMG .onnx files to make them include non-maximum suppression (NMS)
so that NMS is included in exported TensorRT .engine files
"""

import sys
from pathlib import Path

# add repo root filepath dynamically to import modules from other directories 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import onnx
import onnx_graphsurgeon as gs
import numpy as np

def embed_nms(onnx_path, output_path):
    # convert pathlib paths to strings for ONNX compatibility
    onnx_path = str(onnx_path)
    output_path = str(output_path)
    
    graph = gs.import_onnx(onnx.load(onnx_path))
    
    # extract split outputs created during the custom YOLOMG PyTorch export
    boxes_tensor = graph.outputs[0]
    scores_tensor = graph.outputs[1]
    
    # define tensor shapes expected by TensorRT
    num_detections = gs.Variable(name = "num_detections", dtype = np.int32, shape = (1, 1))
    detection_boxes = gs.Variable(name = "detection_boxes", dtype = np.float32, shape = (1, 300, 4))
    detection_scores = gs.Variable(name = "detection_scores", dtype = np.float32, shape = (1, 300))
    detection_classes = gs.Variable(name = "detection_classes", dtype = np.int32, shape = (1, 300))
    
    # create TensorRT plugin node
    nms_node = gs.Node(op = "EfficientNMS_TRT",
                       name = "batched_nms",
                       inputs = [boxes_tensor, scores_tensor],
                       outputs = [num_detections, detection_boxes, detection_scores, detection_classes],
                       attrs = {"plugin_version": "1",
                                "background_class": -1,  # YOLOv5 has no background class
                                "max_output_boxes": 300, # matches YOLOv5 max_det default
                                "score_threshold": 0.001, # keep consistent with existing experiments
                                "iou_threshold": 0.4, # keep consistent with existing experiments
                                "score_activation": False, # sigmoid already applied in yolo.py
                                "box_coding": 1}) # 1 = BoxCenterSize, which natively decodes YOLO xywh to xyxy
    
    graph.nodes.append(nms_node)
    graph.outputs = [num_detections, detection_boxes, detection_scores, detection_classes]
    
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), output_path)
    print(f"Saved NMS-embedded ONNX to {output_path}")

if __name__ == "__main__":
    input_onnx = ROOT / "weights" / "best_640.onnx"
    output_onnx = ROOT / "weights" / "best_640_nms.onnx"
    
    # verify the input file exists
    if not input_onnx.exists():
        print(f"Error: Could not find input ONNX file at {input_onnx}")
        sys.exit(1)
        
    print(f"Loading base ONNX graph from: {input_onnx}")
    embed_nms(input_onnx, output_onnx)