import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os
import shutil
import requests
import time
import pandas as pd

# ---------------- COCO LABELS ----------------
COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

MODEL_URL = "https://huggingface.co/RyanR0512/Yolov5m-tflite/resolve/main/yolov5m-fp16.tflite"
MODEL_PATH = "yolov5m-fp16.tflite"

# ---------------- DOWNLOAD MODEL ----------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... (one-time setup)"):
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

# ---------------- LOAD MODEL ONCE ----------------
@st.cache_resource
def load_model():
    download_model()
    interpreter = tf.lite.Interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# ---------------- NMS HELPERS ----------------
def compute_iou(box1, boxes):
    x1, y1, x2, y2 = box1
    xx1 = np.maximum(x1, boxes[:, 0])
    yy1 = np.maximum(y1, boxes[:, 1])
    xx2 = np.minimum(x2, boxes[:, 2])
    yy2 = np.minimum(y2, boxes[:, 3])

    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)

def non_max_suppression(dets, iou_threshold=0.5):
    if not dets:
        return []

    boxes = np.array([d["bbox"] for d in dets])
    scores = np.array([d["score"] for d in dets])
    classes = np.array([d["class_id"] for d in dets])

    keep = []
    for c in np.unique(classes):
        idxs = np.where(classes == c)[0]
        sorted_idxs = idxs[np.argsort(-scores[idxs])]

        while len(sorted_idxs):
            best = sorted_idxs[0]
            keep.append(best)
            if len(sorted_idxs) == 1:
                break
            ious = compute_iou(boxes[best], boxes[sorted_idxs[1:]])
            sorted_idxs = sorted_idxs[1:][ious < iou_threshold]

    return [dets[i] for i in keep]

# ---------------- DETECTION (OPTIMIZED) ----------------
def run_detection(img_bytes, image_name, interpreter, conf_thresh=0.7, img_dir="dataset/images", lbl_dir="dataset/labels"):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (640, 640))
    h, w, _ = img_resized.shape

    inp = np.expand_dims(
        cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
        0
    )

    # Use pre-loaded interpreter
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], inp)
    interpreter.invoke()

    output = interpreter.get_tensor(
        interpreter.get_output_details()[0]["index"]
    )[0]

    detections = []
    for i, det in enumerate(output):
        cx, cy, bw, bh = det[:4]
        conf = det[4]
        probs = det[5:]
        cls_id = int(np.argmax(probs))
        score = conf * probs[cls_id]

        if score < conf_thresh:
            continue

        cx, cy, bw, bh = cx*w, cy*h, bw*w, bh*h
        x1, y1 = int(cx - bw/2), int(cy - bh/2)
        x2, y2 = int(cx + bw/2), int(cy + bh/2)

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class_id": cls_id,
            "score": score,
            "index": i
        })

    detections = non_max_suppression(detections)

    base = os.path.splitext(image_name)[0]
    saved_classes = []
    preview_path = None

    # Batch file writes
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = img_resized[max(0,y1):y2, max(0,x1):x2]

        img_name = f"{base}_{det['index']}.jpg"
        lbl_name = f"{base}_{det['index']}.txt"

        cv2.imwrite(os.path.join(img_dir, img_name), crop)

        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        with open(os.path.join(lbl_dir, lbl_name), "w") as f:
            f.write(f"{det['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        saved_classes.append(det["class_id"])

    # Generate preview (save to disk instead of keeping in memory)
    preview = img_resized.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0,255,0), 2)
    
    preview_path = os.path.join(img_dir, f"{base}_preview.jpg")
    cv2.imwrite(preview_path, preview)

    return preview_path, saved_classes

# ---------------- STREAMLIT UI ----------------
st.title("📦 YOLO Dataset Builder (Bulk Images)")
st.markdown("**Optimized for large batches** - Model loads once, efficient file I/O")

uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

conf_thresh = st.slider("Confidence Threshold", 0.1, 0.95, 0.7)

show_previews = st.checkbox(
    "Show annotated image previews (may slow down display for large batches)",
    value=False
)

if uploaded_files and st.button("Run Detection & Build Dataset"):
    shutil.rmtree("dataset", ignore_errors=True)
    
    # Create directories once
    img_dir = os.path.join("dataset", "images")
    lbl_dir = os.path.join("dataset", "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # Load model ONCE (cached)
    interpreter = load_model()

    progress = st.progress(0)
    status = st.empty()
    eta = st.empty()

    class_counts = {i: 0 for i in range(len(COCO_CLASSES))}
    preview_paths = [] if show_previews else None

    total = len(uploaded_files)
    start = time.time()

    for i, file in enumerate(uploaded_files, start=1):
        status.text(f"Processing {i}/{total}: {file.name}")

        # Read file bytes once
        img_bytes = file.read()
        
        # Pass pre-loaded interpreter
        preview_path, classes = run_detection(
            img_bytes, 
            file.name, 
            interpreter, 
            conf_thresh,
            img_dir,
            lbl_dir
        )

        if show_previews:
            preview_paths.append((file.name, preview_path))

        for c in classes:
            class_counts[c] += 1

        elapsed = time.time() - start
        avg = elapsed / i
        remaining = avg * (total - i)

        progress.progress(i / total)
        eta.text(f"Estimated time remaining: {remaining:.1f} seconds")

    progress.progress(1.0)
    status.text("Processing complete ✅")
    eta.text("Estimated time remaining: 0 seconds")

    st.subheader("📊 Class-wise Object Counts")
    rows = [
        {"Class": COCO_CLASSES[k], "Count": v}
        for k, v in class_counts.items() if v > 0
    ]
    df = pd.DataFrame(rows).sort_values("Count", ascending=False)
    st.dataframe(df, use_container_width=True)

    if show_previews and preview_paths:
        st.subheader("Annotated Previews")
        for name, path in preview_paths:
            st.markdown(f"**{name}**")
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)

    shutil.make_archive("dataset_export", "zip", "dataset")
    with open("dataset_export.zip", "rb") as f:
        st.download_button(
            "Download Dataset ZIP",
            f,
            file_name="dataset.zip",
            mime="application/zip"
        )

    st.success("Dataset exported in YOLO format")
