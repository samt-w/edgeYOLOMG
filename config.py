# config.py
import os
from pathlib import Path
from dotenv import load_dotenv
import multiprocessing

load_dotenv()

RANDOM_SEED = 42

total_cores = multiprocessing.cpu_count()
# setting this so multiprocessing does not use more than 80% of available cores
RECOMMENDED_CORES = int(total_cores * 0.8)

### DIRECTORY FILEPATHS FOR DATASETS

# Raw Data Paths
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR"))
TRAIN_VIDEOS_DIR = RAW_DATA_DIR / "train_videos"
TEST_VIDEOS_DIR = RAW_DATA_DIR / "test_videos"
ANNOTATIONS_DIR = RAW_DATA_DIR / "annotations"

# Processed Data Paths
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR"))

# Modality directories
IMAGES_DIR = PROCESSED_DATA_DIR / "images"
MASKS_DIR = PROCESSED_DATA_DIR / "masks"
LABELS_DIR = PROCESSED_DATA_DIR / "labels"

# Train splits
IMAGES_TRAIN_DIR = IMAGES_DIR / "train"
MASKS_TRAIN_DIR = MASKS_DIR / "train"
LABELS_TRAIN_DIR = LABELS_DIR / "train"

# Val splits
IMAGES_VAL_DIR = IMAGES_DIR / "val"
MASKS_VAL_DIR = MASKS_DIR / "val"
LABELS_VAL_DIR = LABELS_DIR / "val"

# Test splits
IMAGES_TEST_DIR = IMAGES_DIR / "test"
MASKS_TEST_DIR = MASKS_DIR / "test"
LABELS_TEST_DIR = LABELS_DIR / "test"

### DATASET TRAIN/TEST SPLITS

## ARD100
ARD100_TRAIN_LIST = ['phantom09', 'phantom10', 'phantom14', 'phantom17', 'phantom19', 'phantom20', 'phantom28', 'phantom29', 'phantom30', 'phantom32',
                     'phantom36', 'phantom40', 'phantom42', 'phantom43', 'phantom44', 'phantom46', 'phantom63', 'phantom65', 'phantom66', 'phantom68',
                     'phantom70', 'phantom71', 'phantom74', 'phantom75', 'phantom76', 'phantom77', 'phantom78', 'phantom80', 'phantom81', 'phantom82',
                     'phantom84', 'phantom85', 'phantom86', 'phantom87', 'phantom89', 'phantom90', 'phantom101', 'phantom103', 'phantom104', 'phantom105',
                     'phantom106', 'phantom107', 'phantom108', 'phantom109', 'phantom111', 'phantom112', 'phantom114', 'phantom115', 'phantom116', 'phantom117',
                     'phantom118', 'phantom120', 'phantom132', 'phantom137', 'phantom138', 'phantom139', 'phantom140', 'phantom142', 'phantom143', 'phantom145',
                     'phantom146', 'phantom147', 'phantom148', 'phantom149', 'phantom150']

ARD100_TEST_LIST = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom08', 'phantom22', 'phantom39',
                    'phantom41', 'phantom45', 'phantom47', 'phantom50', 'phantom54', 'phantom55', 'phantom56',
                    'phantom57', 'phantom58', 'phantom60', 'phantom61', 'phantom64', 'phantom73', 'phantom79',
                    'phantom92', 'phantom93', 'phantom94', 'phantom95', 'phantom97', 'phantom102', 'phantom110',
                    'phantom113', 'phantom119', 'phantom133', 'phantom135', 'phantom136', 'phantom141', 'phantom144']

# domain adaptation
# new scenes
sets_new_scenes = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom47', 'phantom50',
                   'phantom54', 'phantom55', 'phantom56', 'phantom57', 'phantom58', 'phantom60']

# low light adaptation
sets_low = ['phantom95', 'phantom97', 'phantom133', 'phantom135', 'phantom136']

small_num = 0

# different size test
set_es = ['phantom04', 'phantom22', 'phantom39', 'phantom41', 'phantom45', 'phantom50', 'phantom54', 'phantom55', 'phantom61', 'phantom64', 'phantom73', 'phantom94']  # smaller than 144
set_rs = ['phantom02', 'phantom56', 'phantom57', 'phantom58', 'phantom60', 'phantom79', 'phantom92', 'phantom102', 'phantom110', 'phantom113', 'phantom119', 'phantom141', 'phantom144']  # 144~400
set_gs = ['phantom03', 'phantom05', 'phantom47', 'phantom93']  # 400~1024

## NPS
NPS_TRAIN_LIST = ['Clip_01', 'Clip_02', 'Clip_03', 'Clip_04', 'Clip_05', 'Clip_06', 'Clip_07', 'Clip_08', 'Clip_09', 'Clip_10',
                  'Clip_11', 'Clip_12', 'Clip_13', 'Clip_14', 'Clip_15', 'Clip_16', 'Clip_17', 'Clip_18', 'Clip_19', 'Clip_20',
                  'Clip_21', 'Clip_22', 'Clip_23', 'Clip_24', 'Clip_25', 'Clip_26', 'Clip_27', 'Clip_28', 'Clip_29', 'Clip_30',
                  'Clip_31', 'Clip_32', 'Clip_33', 'Clip_34', 'Clip_35', 'Clip_36', 'Clip_37', 'Clip_38', 'Clip_39', 'Clip_40']

NPS_TEST_LIST = ['Clip_41', 'Clip_42', 'Clip_43', 'Clip_44', 'Clip_45', 'Clip_46', 'Clip_47', 'Clip_48', 'Clip_49', 'Clip_50']