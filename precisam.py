import streamlit as st
from sam2.sam2_image_predictor import SAM2ImagePredictor
import zipfile
import json
import os
import cv2
import torch
import numpy as np
from copy import deepcopy
import shutil
import atexit


# Define the cleanup function to be called on program exit
def cleanup():
    folder_to_remove = "output"
    if os.path.exists(folder_to_remove):
        shutil.rmtree(folder_to_remove)


# Register the cleanup function to be called on program exit
atexit.register(cleanup)


# Function to recalculate bboxes and areas
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


def initialize_model(selected_model):
    return SAM2ImagePredictor.from_pretrained(selected_model)


def bbox_to_xyxy(bbox):
    return np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])


def find_first_json(path_dir_annotations):
    # Cherche tous les fichiers JSON dans le dossier annotations
    json_files = [f for f in os.listdir(path_dir_annotations) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError("No JSON files found in annotations folder.")
    return os.path.join(path_dir_annotations, json_files[0])


def adjust_image_path(path_dir_images):
    # List all the contents in the 'images' directory
    contents = os.listdir(path_dir_images)

    # Check if there is exactly one item and if it is a directory
    if len(contents) == 1 and os.path.isdir(os.path.join(path_dir_images, contents[0])):
        # Redirect to this subdirectory
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
                # Adds the file to the ZIP with its path relative to the base folder
                arcname = os.path.relpath(file_path, directory_path)
                zip_file.write(file_path, arcname)
    
    return zip_path


def main():
    st.title("Bounding Box Recalibration Application with SAM")
    st.write("This application recalculates annotation bounding boxes using the SAM model.")

    models = ["facebook/sam2-hiera-large", "facebook/sam2-hiera-small", "facebook/sam2-hiera-tiny", "facebook/sam2-hiera-base-plus"]
    selected_model = st.selectbox("Select a model", models)

    # Initialize session state variables
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'process_started' not in st.session_state:
        st.session_state.process_started = False
    if 'download_ready' not in st.session_state:
        st.session_state.download_ready = False
    if 'new_zip_path' not in st.session_state:
        st.session_state.new_zip_path = None

    # File uploader for the ZIP archive
    uploaded_file = st.file_uploader("Upload a ZIP file containing annotations and images", type=["zip"])

    # Update session state when a file is uploaded
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.process_started = False
        st.session_state.download_ready = False
        st.session_state.new_zip_path = None

    # Button to start the process
    if st.button("Start Process"):
        st.session_state.process_started = True
        if st.session_state.uploaded_file is None:
            st.error("Please upload a ZIP file before starting the process.")
            st.session_state.process_started = False
            st.session_state.download_ready = False
            st.session_state.new_zip_path = None
        else:
            with st.spinner("Initializing model..."):
                model = initialize_model(selected_model)

            with st.spinner("Processing..."):
                zip_name = os.path.splitext(st.session_state.uploaded_file.name)[0]
                extract_dir = os.path.join("extracted_data", zip_name)
                os.makedirs(extract_dir, exist_ok=True)
                
                with zipfile.ZipFile(st.session_state.uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                path_dir_annotations = os.path.join(extract_dir, "annotations")
                path_dir_images = adjust_image_path(os.path.join(extract_dir, "images"))
                annotation_file = find_first_json(path_dir_annotations)

                with open(annotation_file, "r") as f:
                    json_annotation = json.load(f)
                images_dict = {image["id"]: image for image in json_annotation["images"]}
                json_annotation_modified = deepcopy(json_annotation)

                for annotation in json_annotation_modified["annotations"]:
                    if "sam_bbox" in annotation and annotation["sam_bbox"] == 1:
                        continue
                    else:
                        image_id = annotation["image_id"]
                        image_name = images_dict.get(image_id)["file_name"]
                        image = cv2.cvtColor(cv2.imread(os.path.join(path_dir_images, image_name)), cv2.COLOR_BGR2RGB)

                        bbox = annotation["bbox"]
                        new_bbox, new_area = sam_bbox(bbox, image, model)
                        annotation["bbox"] = new_bbox
                        annotation["area"] = new_area
                        annotation["sam_bbox"] = 1

                with open(annotation_file, "w") as f:
                    json.dump(json_annotation_modified, f)

            new_zip_name = f"{zip_name}_precisam.zip"
            new_zip_path = create_zip_with_updated_json(extract_dir, new_zip_name, "output")
            shutil.rmtree("extracted_data")

            st.session_state.download_ready = True
            st.session_state.new_zip_path = new_zip_path
            st.success("Processing complete! The output ZIP is ready for download.")


    # Provide a download link for the new ZIP file
    if st.session_state.download_ready and st.session_state.new_zip_path:
        with open(st.session_state.new_zip_path, "rb") as f:
            st.download_button(
                label="Download processed ZIP",
                data=f,
                file_name=os.path.basename(st.session_state.new_zip_path),
                mime="application/zip"
            )


if __name__ == "__main__":
    main()