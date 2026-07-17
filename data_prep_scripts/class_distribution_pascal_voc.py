"""
code for visualising the class count across the target dataset - assuming labels in Pascal VOC format

script also saves class count data in a dynamically named directory at same root as script

target directories are typically: ./<dataset_name>/annotations/<video_name>/<video_frame>.xml
"""
import os
from collections import Counter
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
import json
import xml.etree.ElementTree as ET
# annotations filepath
from config import RAW_DATA_DIR, ANNOTATIONS_DIR, RECOMMENDED_CORES

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
    
    return frame_object_counts, class_counts

def pascal_voc_object_count(parent_folder: str) -> tuple:
    # review all .xml files in the subfolders of a given folder, and return two dictionaries:
    # - one with number of objects in each .xml frame in the directory
    # - one with number of objects in each class across entire dataset
    
    # initialise counters
    total_frame_object_counts = Counter({"None": 0,
                                   "Single Object": 0,
                                   "Multiple Objects": 0})
    total_class_counts = Counter()
    
    # generate list of folder paths
    folders = [folder.path for folder in os.scandir(parent_folder) if folder.is_dir()]
    
    # distribute folders across multiple processes
    with concurrent.futures.ProcessPoolExecutor(max_workers = RECOMMENDED_CORES) as executor:
        results = executor.map(process_single_folder, folders)
        # aggregate results across all workers
        for folder_frame_object_counts, folder_class_counts in results:
            total_frame_object_counts.update(folder_frame_object_counts)
            total_class_counts.update(folder_class_counts)
    
    return dict(total_frame_object_counts), dict(total_class_counts)

if __name__ == "__main__":
    object_count_dict, object_class_count_dict = pascal_voc_object_count(ANNOTATIONS_DIR)

    print("Frame Counts:", object_count_dict)
    print("Class Counts:", object_class_count_dict)

    # Convert to DataFrames, orienting keys as the index
    frame_object_count_df = pd.DataFrame.from_dict(object_count_dict, orient='index', columns=['Count'])
    object_class_count_df = pd.DataFrame.from_dict(object_class_count_dict, orient='index', columns=['Count'])

    # Visualise categorical data using bar charts
    frame_object_count_plot = frame_object_count_df.plot.bar(title="Number of objects in each Frame")
    class_count_plot = object_class_count_df.plot.bar(title="Total Class Counts")

    ### save data to directory

    # directory name
    dataset_name = os.path.basename(os.path.normpath(RAW_DATA_DIR))
    save_dir = f"{dataset_name}_class_counts"
    # initialise directory
    os.makedirs(save_dir, exist_ok = True)
    
    # write class count dictionaries to .json
    with open(os.path.join(save_dir, "total_frame_object_count.json"), "w") as fp:
        json.dump(object_count_dict, fp, sort_keys = True, indent = 4)
    with open(os.path.join(save_dir, "total_dataset_class_count.json"), "w") as fp:
        json.dump(object_class_count_dict, fp, sort_keys = True, indent = 4)

    frame_object_count_plot.get_figure().savefig(os.path.join(save_dir, "total_frame_object_count.png"))
    class_count_plot.get_figure().savefig(os.path.join(save_dir, "total_dataset_class_count.png"))

    # Display the plots
    plt.show()