"""
code for visualising the class count across the target dataset

target directories are typically: ./annotations/<video_name>/<video_frame>.xml
"""
import os
from collections import Counter
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
# annotations filepath
from config import ANNOTATIONS_DIR, RECOMMENDED_CORES

def process_single_folder(folder_path: str) -> tuple:
    # worker function to process .xml files in a single folder
    
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
    frame_object_count_df.plot.bar(title="Number of objects in each Frame")
    object_class_count_df.plot.bar(title="Total Class Counts")
    
    # Display the plots
    plt.show()