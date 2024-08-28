import streamlit as st
import os
import zipfile
import shutil
import json
import atexit
from utils.utils import initialize_model, adjust_image_path, find_first_json, create_zip_with_updated_json, process_annotations, cleanup


atexit.register(cleanup)


def main():
    st.title("Bounding Box Recalibration Application with SAM")
    st.write("This application recalculates annotation bounding boxes using the SAM model.")

    models = ["facebook/sam2-hiera-large", "facebook/sam2-hiera-small", "facebook/sam2-hiera-tiny", "facebook/sam2-hiera-base-plus"]
    selected_model = st.selectbox("Select a model", models)

    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'process_started' not in st.session_state:
        st.session_state.process_started = False
    if 'download_ready' not in st.session_state:
        st.session_state.download_ready = False
    if 'new_zip_path' not in st.session_state:
        st.session_state.new_zip_path = None

    uploaded_file = st.file_uploader("Upload a ZIP file containing annotations and images", type=["zip"])

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.process_started = False
        st.session_state.download_ready = False
        st.session_state.new_zip_path = None

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

                json_annotation_modified = process_annotations(annotation_file, path_dir_images, model)

                with open(annotation_file, "w") as f:
                    json.dump(json_annotation_modified, f)

            new_zip_name = f"{zip_name}_precisam.zip"
            new_zip_path = create_zip_with_updated_json(extract_dir, new_zip_name, "output")
            shutil.rmtree("extracted_data")

            st.session_state.download_ready = True
            st.session_state.new_zip_path = new_zip_path
            st.success("Processing complete! The output ZIP is ready for download.")

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
