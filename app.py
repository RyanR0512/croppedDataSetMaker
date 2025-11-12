import os
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from croppedDataSetMaker import run_detection

# -------------------- Your Detection Functions --------------------
# Include all your detection functions here: preprocess_image, compute_iou,
# non_max_suppression, download_model, run_detection (from the last script)
# Make sure run_detection saves crops and returns detection count.

# For brevity, assuming run_detection is imported or defined above

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="Object Crop Dataset Builder", layout="wide")
st.title("📸 Object Crop Dataset Builder")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

output_dir = "dataset_crops"
os.makedirs(output_dir, exist_ok=True)

if uploaded_file:
    # Convert uploaded file to OpenCV image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Save original upload for reference
    input_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    image.save(input_path)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.info("Running object detection and cropping...")

    # Run detection
    num_crops = run_detection(input_path, output_dir=output_dir)

    st.success(f"✅ Detected {num_crops} objects and saved crops to '{output_dir}'")

    # Show cropped images in the app
    crop_files = sorted(os.listdir(output_dir))
    if crop_files:
        st.subheader("Cropped Objects")
        cols = st.columns(4)
        for idx, crop_file in enumerate(crop_files[-num_crops:]):  # show only last uploaded
            img_path = os.path.join(output_dir, crop_file)
            img = Image.open(img_path)
            cols[idx % 4].image(img, caption=crop_file, use_column_width=True)
