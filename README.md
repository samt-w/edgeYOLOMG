# edgeYOLOMG

This is an adapted repo, cloned from the [YOLOMG](https://github.com/Irisky123/YOLOMG) repo, for an MSc Project.

This is a model trying to detect UAVs in video data. The original YOLOMG model was adapted from YOLOv5s, adding a motion-masking classical computer vision algorithm.

This project aims to run YOLOMG on edge hardware (an NVIDIA Jetson Orin Nano) to see whether the algorithm's high mAP and FPS can be maintained on constrained hardware.

Full step-by-step instructions (including environment setup, RunPod cloud training, and Jetson-specific commands) are in the project report's software appendix. This README is a quick-reference summary.

## Dataset

The dataset is the ARD100 dataset, created by the original researchers:
- [BaiduYun](https://pan.baidu.com/s/1ycAoKbzQ1rlzvKr8VRakgw?pwd=1x2z) (code: 1x2z)

![Dataset Example Images](data/ARD100_samples_show.png "Example Images")

## Repo Structure

```
edgeYOLOMG/
├── cloud/                 # scripts for running training in the cloud (RunPod)
├── data/                  # dataset config, PTQ calibration set scripts
├── data_prep_scripts/     # scripts for preprocessing/labelling the dataset
├── edge/                  # export, PTQ, and inference scripts for the Jetson/host
├── models/                # network architecture .yaml files and core PyTorch modules
├── utils/                 # YOLO dependencies (dataloaders, loss functions, metrics)
└── weights/               # trained .pt/.onnx/.engine model weights
```

`.env.sanitised` lists the environment variables (filepaths and API credentials) needed to run the code. It must be copied to `.env` and completed before running any code. `config.py` reads `.env` and derives the filepaths used throughout the pipeline.

## 1. Data Extraction and Preprocessing

Converts raw ARD100 videos/annotations into RGB frame, motion-mask, and YOLO-label triplets.

```bash
pip install -r requirements.txt
cd data_prep_scripts
python data_prep_pipeline_test.py   # quick end-to-end check (6 videos)
python data_prep_pipeline_full.py   # full dataset
```

## 2. Model Training (Cloud)

Trains on a RunPod GPU pod. Assumes a Hopper-architecture GPU (e.g. H100).

```bash
tmux new -s train
python train_cloud.py --data data/ARD100_1280.yaml --cfg models/YOLOMG_ARD100.yaml \
    --batch-size -1 --epochs 2 --imgsz 1280 --name ARD100-test-1280-
```
`train_cloud.py` extracts the preprocessed dataset from a network-volume `.tar` archive, logs to ClearML, and terminates the pod on completion.

## 3. Inference on Host Device

Runs the full detection pipeline on a machine with a discrete GPU, reporting mAP and FPS.

```bash
cd edge
python inference_host.py
```
Configure the necessary arguments (e.g. `weights`, `imgsz`, `video_dir`) at the bottom of the script (`if __name__ == "__main__":`) before running.

## 4. Inference on Edge Device (Jetson)

Runs the TensorRT-compiled model on a Jetson Orin Nano Super (L4T 36.5.2 / JetPack 6.2.3, `jetson-containers` with the `l4t-ml:36.4.0` image).

```bash
# Jetson host terminal (outside the container):
sudo jetson_clocks
jetson-containers run -d --name yolomg \
    -v /ssd/workspace/yolomg:/workspace/yolomg -v /ssd/workspace/videos:/workspace/videos \
    $(autotag l4t-ml)
docker exec -it yolomg bash

# container terminal:
cd workspace/yolomg
pip install -r requirements-jetson.txt --index-url https://pypi.org/simple
python3 edge/inference_jetson.py              # accuracy + FPS (low conf. threshold)
# or
python3 edge/inference_jetson_deployment.py   # realistic deployment FPS
```
As with `inference_host.py`, run parameters must be configured at the bottom of each script.

## 5. Post-Training Quantisation (Edge)

Compiles INT8 TensorRT engines for the Jetson, calibrated on a stratified sample of training frames.

```bash
# build calibration set (repo root)
python3 data/ptq_extract_tensors.py

# export ONNX
python3 edge/export_YOLOMG.py --weights weights/best_640.pt --imgsz 640 --include onnx

# compile INT8 engine (on the Jetson, inside the container — see Section 4)
tmux new -s engine
python3 edge/ptq_compile_int8_engine.py \
    --onnx weights/best_640.onnx --engine weights/best_640_int8_fp16model6.engine \
    --rgb_dir data/ptq_calibration_RGB_tensors_640 --mask_dir data/ptq_calibration_MASKS_tensors_640 \
    --imgsz 640 --batch 1 --strategy fp16model6
```
`--strategy fp16model6` pins the model's backbone and detection heads to FP16 (leaving the neck at INT8), giving the best accuracy/speed trade-off found in this project. The resulting `.engine` file can be substituted into `inference_jetson.py` in place of the `.pt` weights.
