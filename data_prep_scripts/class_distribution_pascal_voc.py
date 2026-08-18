"""
code for visualising the class count across the target dataset - assuming labels in Pascal VOC format

code assesses class counts across the train and test splits

script also saves class count data in a dynamically named directory at same root as script

target dataset directories are typically: ./<dataset_name>/annotations/<video_name>/<video_name>_<frame_number>.xml
"""
import os
from collections import Counter
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
import json
import xml.etree.ElementTree as ET
from config import RAW_DATA_DIR, ANNOTATIONS_DIR, RECOMMENDED_CORES, ARD100_TRAIN_LIST, ARD100_TEST_LIST

# convert lists to sets for efficient lookups
TRAIN_SET = set(ARD100_TRAIN_LIST)
TEST_SET = set(ARD100_TEST_LIST)

def process_single_folder(folder_path: str) -> tuple:
    # worker function to process .xml files in a single folder

    # print update to console
    print(f"Now processing item {folder_path}.")
    
    # initialise counters
    frame_object_counts = Counter({"None": 0,
                                   "Single Object": 0,
                                   "Multiple Objects": 0})
    class_counts = Counter()

    # iterate through files in the folder
    for file in os.scandir(folder_path):
        # skip any non-XML files
        if not file.name.endswith(".xml"):
            continue

        try:
            # parse XML and extract object names:
            tree = ET.parse(file.path)
            frame_objects = [obj.findtext("name") for obj in tree.iter("object")]

            # report non-drone classes
            for obj_name in frame_objects:
                if obj_name != "Drone":
                    print(f"Class '{obj_name}' found in {file.path}")

            # update object counts
            num_objects = len(frame_objects)
            if num_objects == 0:
                frame_object_counts["None"] += 1
            elif num_objects == 1:
                frame_object_counts["Single Object"] += 1
            else:
                frame_object_counts["Multiple Objects"] += 1
            
            class_counts.update(frame_objects)
        
        except ET.ParseError:
            # ignore corrupted/empty XML files
            pass
    
    # return folder name with counts, for train/test split matching
    return os.path.basename(folder_path), frame_object_counts, class_counts

def pascal_voc_object_count(parent_folder: str) -> tuple:
    # review all .xml files in the subfolders of a given folder, and return two dictionaries:
    # - one with number of objects in each .xml frame in the directory
    # - one with number of objects in each class across entire dataset
    
    # initialise counters for entire dataset and train/test splits
    total_frame_object_counts = Counter({"None": 0,
                                   "Single Object": 0,
                                   "Multiple Objects": 0})
    total_class_counts = Counter()

    train_frame_object_counts = Counter({"None": 0,
                                   "Single Object": 0,
                                   "Multiple Objects": 0})
    train_class_counts = Counter()

    test_frame_object_counts = Counter({"None": 0,
                                   "Single Object": 0,
                                   "Multiple Objects": 0})
    test_class_counts = Counter()
    
    # generate list of folder paths
    folders = [folder.path for folder in os.scandir(parent_folder) if folder.is_dir()]
    
    # distribute folders across multiple processes
    with concurrent.futures.ProcessPoolExecutor(max_workers = RECOMMENDED_CORES) as executor:
        results = executor.map(process_single_folder, folders)
        # aggregate results across all workers
        for folder_name, folder_frame_object_counts, folder_class_counts in results:
            total_frame_object_counts.update(folder_frame_object_counts)
            total_class_counts.update(folder_class_counts)

            if folder_name in TRAIN_SET:
                train_frame_object_counts.update(folder_frame_object_counts)
                train_class_counts.update(folder_class_counts)
            
            elif folder_name in TEST_SET:
                test_frame_object_counts.update(folder_frame_object_counts)
                test_class_counts.update(folder_class_counts)
    
    return (
        dict(total_frame_object_counts), dict(total_class_counts),
        dict(train_frame_object_counts), dict(train_class_counts),
        dict(test_frame_object_counts), dict(test_class_counts),
    )

if __name__ == "__main__":
    (total_object_count_dict, total_object_class_count_dict,
     train_object_count_dict, train_object_class_count_dict,
     test_object_count_dict, test_object_class_count_dict) = pascal_voc_object_count(ANNOTATIONS_DIR)

    print("Total Frame Counts:", total_object_count_dict)
    print("Total Class Counts:", total_object_class_count_dict)
    print("-" * 20)
    print("Train Frame Counts:", train_object_count_dict)
    print("Train Class Counts:", train_object_class_count_dict)
    print("-" * 20)
    print("Test Frame Counts:", test_object_count_dict)
    print("Test Class Counts:", test_object_class_count_dict)
    print("-" * 20)

    ### save data to directory

    # directory name
    dataset_name = os.path.basename(os.path.normpath(RAW_DATA_DIR))
    save_dir = f"{dataset_name}_class_counts"
    # initialise directory
    os.makedirs(save_dir, exist_ok = True)

    # group splits into dictionary for processing
    splits_data = {
        "total": (total_object_count_dict, total_object_class_count_dict),
        "train": (train_object_count_dict, train_object_class_count_dict),
        "test": (test_object_count_dict, test_object_class_count_dict)
    }

    # iterate through each split to generate pandas dataframes, plots, and .json
    for split_prefix, (frame_dict, class_dict) in splits_data.items():
        
        # write class count dictionaries to .json
        with open(os.path.join(save_dir, f"{split_prefix}_object_count.json"), "w") as fp:
            json.dump(frame_dict, fp, sort_keys = True, indent = 4)
        with open(os.path.join(save_dir, f"{split_prefix}_dataset_class_count.json"), "w") as fp:
            json.dump(class_dict, fp, sort_keys = True, indent = 4)

        # convert to dataframes, orienting keys as the index
        frame_object_count_df = pd.DataFrame.from_dict(frame_dict, orient='index', columns=['Count'])
        object_class_count_df = pd.DataFrame.from_dict(class_dict, orient='index', columns=['Count'])

        # visualise data using bar plots
        frame_object_count_plot = frame_object_count_df.plot.bar(title=f"{split_prefix.upper()} - Objects per Frame", rot=0)
        class_count_plot = object_class_count_df.plot.bar(title=f"{split_prefix.upper()} - Total Class Counts", rot=0)

        # add count labels to bar plots
        frame_object_count_plot.bar_label(frame_object_count_plot.containers[0])
        class_count_plot.bar_label(class_count_plot.containers[0])

        # save plots to directory
        frame_object_count_plot.get_figure().savefig(os.path.join(save_dir, f"{split_prefix}_frame_object_count.png"), bbox_inches="tight")
        class_count_plot.get_figure().savefig(os.path.join(save_dir, f"{split_prefix}_dataset_class_count.png"), bbox_inches="tight")

    # Display the plots
    plt.show()