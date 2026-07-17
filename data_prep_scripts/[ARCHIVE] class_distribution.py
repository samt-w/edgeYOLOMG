"""
code for visualising the class count across the target dataset

target directories are typically: ./annotations/<video_name>/<video_frame>.xml
"""
import os
import pandas as pd
import xml.etree.ElementTree as ET
# annotations filepath
from config import ANNOTATIONS_DIR

def return_folders_as_list(directory_path: str) -> list:
    # return list of strings of all folders in parent folder
    return [str(folder.name) for folder in os.scandir(directory_path) if folder.is_dir()]

def extract_xml_filepaths(folder_path: str) -> list:
    # return list of all .xml files in folder
    return [str(file.name) for file in os.scandir(folder_path) if str(file.name).endswith(".xml")]

# define function to read Pascal VOC XML files
def read_xml_content(xml_filepath: str) -> dict:
    """
    reads a .xml Pascal VOC file and returns a dictionary with class counts for objects in that frame
    """
    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    object_num = 0

    class_count_dict = {}

    for object in root.iter("object"):
        object_num += 1
        object_name = object.find("name").text
        if object_name in class_count_dict:
            class_count_dict[object_name] += 1
        else:
            class_count_dict[object_name] = 1

    if object_num == 0:
        return {"None": 1}

    return class_count_dict

def pascal_voc_object_count(parent_folder: str) -> dict:
    # review all .xml files in given folder, and return two dictionaries:
    # - one with number of objects in each .xml frame in the directory
    # - one with number of objects in each class across entire dataset
    object_count_dict = {"None": 0,
                         "Single Object": 0,
                         "Multiple Objects": 0}
    
    object_class_count_dict = {}
    
    folder_list = return_folders_as_list(parent_folder)

    folder_count = 0
    
    for folder in folder_list:
        folder_count += 1
        print(f"Now extracting from folder {folder_count} / 100.")
        xml_list = extract_xml_filepaths(os.path.join(parent_folder, folder))
        for xml_file in xml_list:
            frame_object_dict = read_xml_content(os.path.join(parent_folder, folder, xml_file))
            # update count of number of objects in each frame
            if "None" in frame_object_dict:
                object_count_dict["None"] += 1
            else:
                for item in frame_object_dict:
                    if len(frame_object_dict) == 1:
                        object_count_dict["Single Object"] += 1
                        # update count of different classes in dataset
                        if item not in object_class_count_dict:
                            object_class_count_dict[item] = frame_object_dict[item]
                        else:
                            object_class_count_dict[item] += frame_object_dict[item]
                    else:
                        object_count_dict["Multiple Object"] += 1
                        # update count of different classes in dataset
                        if item not in object_class_count_dict:
                            object_class_count_dict[item] = frame_object_dict[item]
                        else:
                            object_class_count_dict[item] += frame_object_dict[item]

    return object_count_dict, object_class_count_dict

if __name__ == "__main__":
    # initialise list of folders
    folder_list = return_folders_as_list(ANNOTATIONS_DIR)
    print(folder_list[0:5])

    # check .xml strings return
    xml_list = extract_xml_filepaths(os.path.join(ANNOTATIONS_DIR, folder_list[0]))
    print(xml_list[0:5])

    object_count_dict, object_class_count_dict = pascal_voc_object_count(ANNOTATIONS_DIR)

    print(object_count_dict)

    # # initialise data
    # data = []

    # # initialise data as pandas df
    # df = pd.DataFrame(data)
    
    # # visualise pandas dataframe e.g. with df.hist()
    # histogram = df.hist()