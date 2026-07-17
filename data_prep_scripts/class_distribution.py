"""
code for visualising the class count across the target dataset

target directories are typically: ./annotations/<video_name>/<video_frame>.xml
"""
import os
from collections import Counter
import pandas as pd
import xml.etree.ElementTree as ET
# annotations filepath
from config import ANNOTATIONS_DIR

def pascal_voc_object_count(parent_folder: str) -> tuple:
    # review all .xml files in the subfolders of a given folder, and return two dictionaries:
    # - one with number of objects in each .xml frame in the directory
    # - one with number of objects in each class across entire dataset
    
    # initialise counters
    frame_object_counts = Counter({"None": 0,
                                   "Single Object": 0,
                                   "Multiple Objects": 0})
    class_counts = Counter()

    folder_count = 0
    
    # iterate through folders:
    for item in os.scandir(parent_folder):
        # skip anything that isn't a folder
        if not item.is_dir():
            continue

        folder_count += 1
        print(f"extracting objects from files in folder {folder_count} / 100.")

        # iterate through files in the folder
        for file in os.scandir(item.path):
            # skip any non-XML files
            if not file.name.endswith(".xml"):
                continue

            # parse XML and extract object names:
            tree = ET.parse(file.path)
            frame_objects = [object.findtext("name") for object in tree.iter("object")]

            # update object counts
            num_objects = len(frame_objects)
            if num_objects == 0:
                frame_object_counts["None"] += 1
            elif num_objects == 1:
                frame_object_counts["Single Object"] += 1
            else:
                frame_object_counts["Multiple Objects"] += 1

            # update dataset class counts
            class_counts.update(frame_objects)
        
        print(f"After folder {folder_count} / 100, counts are:")
        print("Frame Counts:", frame_object_counts)
        print("Class Counts:", class_counts) 
    
    return dict(frame_object_counts), dict(class_counts)

if __name__ == "__main__":
    object_count_dict, object_class_count_dict = pascal_voc_object_count(ANNOTATIONS_DIR)

    print("Frame Counts:", object_count_dict)
    print("Class Counts:", object_class_count_dict)

    # # initialise data
    # data = []

    # # initialise data as pandas df
    # df = pd.DataFrame(data)
    
    # # visualise pandas dataframe e.g. with df.hist()
    # histogram = df.hist()