# YOLOMG-STW
This is an adapted repo, cloned from the 'YOLOMG' repo, for Sam Taylor-Wilmshurst's MSc project at Birkbeck College. 

This is a model trying to detect UAVs in video data. The original YOLOMG model was adapted from YOLOv5s. It added a motion-masking classical computer vision algorithm.

This project aims to run YOLOMG on edge hardware (a NVIDIA Jetson Orin Nano) to see whether the algorithm's high mAP and FPS can be maintained on constrained hardware.

# Dataset
The dataset is the ARD100 dataset, created by the original researchers, which can be found here:
- [BaiduYun](https://pan.baidu.com/s/1ycAoKbzQ1rlzvKr8VRakgw?pwd=1x2z ) (code:1x2z)

![Dataset Example Images](data/ARD100_samples_show.png "Example Images ")

# Repo Structure

YOLOMG/  
├── data_prep_scripts/         # scripts for generating and formatting the dataset  
├── models/                    # network architecture .yaml documents and core PyTorch modules  
├── test_code/                 # scripts for creating RGB frames and motion masks from videos  
└── utils/                     # core dependencies for YOLO model, such as dataloaders, loss functions, and metrics  

.env.sanitised contains the global variables that need to be completed by the user in order to run the code  
config.py sets the global variables as paths for scripts to use  

## scripts for pre-processing and labelling the datasets are in ./data_prep_scripts/
```python generate_motion_masks.py```
* this is applied to generate the motion masks from the video files

```python YOLOMG_extract_frames.py```
* this is used to extract the RGB image frames

```python generate_dataset.py```
* this is used to generate train/test datasets from the extracted RGB frames and motion masks. It splits the data based on video IDs rather than by random frames.

## data processing in ./data directory
```python3 split_train_val.py --xml_path xx/xxx/Annotations --txt_path xx/xxx/ImageSets/Main```
* this is used to read the XML annotation files and then split the data into training and validation sets. It then saves the resulting list of filenames into the target directory

```python3 voc2yolo.py```
* this converts the PASCAL VOC XML bounding box annotations into YOLO TXT format

```python3 voc_label.py```
* this generates the text files with the file paths for the RGB images (e.g. ...\train.txt, ...\val.txt, ...\test.txt) which the YOLO model will iterate through

```python3 voc_label2.py```
* same as above but for the motion masking images

## executing training, evaluation, and inference
```python3 train.py --data data/NPS.yaml --cfg models/NPS_uav_s.yaml --weights yolov5s.pt --batch-size 8 --epochs 100 --imgsz 1280 --name NPS-1280```
* this runs a training run on a single GPU, using the NPS.yaml configuration, building the NPS_uav_s.yaml model architecture, initialising with standard YOLOv5s weights, and training with 100 epochs on 1280x1280 size images, before saving the outputs in the NPS-1280 directory

```python -m torch.distributed.run --nproc_per_node=4 --master_port 12345 train.py --data data/ARD100_mask32.yaml --cfg models/ARD100_drone_s.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --imgsz 1280 --name ARD100_mask32-1280 --device 0,1,2,3```
* this runs a distributed training run, in this case on four GPUs

```python3 val.py --weights runs/train/NPS-1280/weights/best.pt --data data/NPS_test.yaml --task val --conf-thres 0.001 --name NPS_test-1280 --imgsz 1280 --batch-size 8 --device 0```
* this evaluates the trained best.pt weights at a very low confidence threshold (0.001)

```python3 dualdetector.py```
* this runs a quick inference run to verify the system works
