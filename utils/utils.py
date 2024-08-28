import zipfile
import json
import os
import cv2
import torch
import numpy as np
from copy import deepcopy
import shutil

from sam2.sam2_image_predictor import SAM2ImagePredictor


def initialize_model(selected_model):
    return SAM2ImagePredictor.from_pretrained(selected_model)


def cleanup():
    folders_to_remove = ["output", "extracted_data"]
    for folder in folders_to_remove:
        if os.path.exists(folder):
            shutil.rmtree(folder)


def sam_bbox(bbox, image, model):
    xyxy = bbox_to_xyxy(bbox)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.set_image(image)
        masks, _, _ = model.predict(point_coords=None, point_labels=None, box=xyxy, multimask_output=False)

    rows, cols = np.where(masks[0])
    x_min = np.min(cols)
    y_min = np.min(rows)
    x_max = np.max(cols)
    y_max = np.max(rows)

    box_width = x_max - x_min
    box_height = y_max - y_min

    new_bbox = [float(x_min), float(y_min), float(box_width), float(box_height)]
    new_area = float(box_width * box_height)

    return new_bbox, new_area


def process_annotations(annotation_file, path_dir_images, model):
    with open(annotation_file, "r") as f:
        json_annotation = json.load(f)

    images_dict = {image["id"]: image for image in json_annotation["images"]}
    json_annotation_modified = deepcopy(json_annotation)

    for annotation in json_annotation_modified["annotations"]:
        if "sam_bbox" in annotation and annotation["sam_bbox"] == 1:
            continue
        
        image_id = annotation["image_id"]
        image_name = images_dict.get(image_id)["file_name"]
        image = cv2.cvtColor(cv2.imread(os.path.join(path_dir_images, image_name)), cv2.COLOR_BGR2RGB)
        
        bbox = annotation["bbox"]
        new_bbox, new_area = sam_bbox(bbox, image, model)
        annotation["bbox"] = new_bbox
        annotation["area"] = new_area
        annotation["sam_bbox"] = 1

    return json_annotation_modified


def bbox_to_xyxy(bbox):
    return np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])


def find_first_json(path_dir_annotations):
    json_files = [f for f in os.listdir(path_dir_annotations) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError("No JSON files found in annotations folder.")
    return os.path.join(path_dir_annotations, json_files[0])


def adjust_image_path(path_dir_images):
    contents = os.listdir(path_dir_images)
    if len(contents) == 1 and os.path.isdir(os.path.join(path_dir_images, contents[0])):
        path_dir_images = os.path.join(path_dir_images, contents[0])
    return path_dir_images


def create_zip_with_updated_json(directory_path, zip_name, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    zip_path = os.path.join(output_dir, zip_name)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for foldername, subfolders, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                arcname = os.path.relpath(file_path, directory_path)
                zip_file.write(file_path, arcname)
    
    return zip_path