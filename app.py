import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os
import shutil
import requests
import time
import gc
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

BATCH_SIZE = 10  # Number of images processed before forcing a GC sweep

# ---------------- DOWNLOAD MODEL ----------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (one-time)..."):
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

# ---------------- CACHED INTERPRETER ----------------
# Created ONCE per session — not once per image.
@st.cache_resource
def load_interpreter():
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

# ---------------- DETECTION ----------------
def run_detection(interpreter, img_bytes, image_name, conf_thresh=0.7, output_dataset="dataset"):
    """
    Runs detection on a single image.
    - interpreter is passed in (shared, cached) rather than recreated here.
    - All intermediate numpy arrays are explicitly deleted after use.
    """
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    del arr  # free raw bytes buffer

    img_resized = cv2.resize(img, (640, 640))
    del img  # free original decoded image
    h, w, _ = img_resized.shape

    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = np.expand_dims(rgb, 0)
    del rgb

    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], inp)
    interpreter.invoke()
    del inp

    output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]

    detections = []
    for i, det in enumerate(output):
        cx, cy, bw, bh = det[:4]
        conf = det[4]
        probs = det[5:]
        cls_id = int(np.argmax(probs))
        score = float(conf * probs[cls_id])

        if score < conf_thresh:
            continue

        cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class_id": cls_id,
            "score": score,
            "index": i
        })

    detections = non_max_suppression(detections)

    img_dir = os.path.join(output_dataset, "images")
    lbl_dir = os.path.join(output_dataset, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    base = os.path.splitext(image_name)[0]
    saved_classes = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = img_resized[max(0, y1):y2, max(0, x1):x2]

        img_name = f"{base}_{det['index']}.jpg"
        lbl_name = f"{base}_{det['index']}.txt"

        cv2.imwrite(os.path.join(img_dir, img_name), crop)
        del crop

        cx_n = (x1 + x2) / 2 / w
        cy_n = (y1 + y2) / 2 / h
        bw_n = (x2 - x1) / w
        bh_n = (y2 - y1) / h

        with open(os.path.join(lbl_dir, lbl_name), "w") as f:
            f.write(f"{det['class_id']} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}\n")

        saved_classes.append(det["class_id"])

    # Build preview and free the resized image
    preview = None
    if img_resized is not None:
        preview = img_resized.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        del img_resized

    return preview, saved_classes


# ---------------- STREAMLIT UI ----------------
st.title("Cropped Dataset Builder YOLO Format (Bulk Images)")

uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

conf_thresh = st.slider("Confidence Threshold", 0.1, 0.95, 0.7)

show_previews = st.checkbox(
    "Show annotated image previews (may slow down large batches)",
    value=False
)

if uploaded_files and st.button("Run Detection & Build Dataset"):
    shutil.rmtree("dataset", ignore_errors=True)

    # Load interpreter once for the entire run
    interpreter = load_interpreter()

    progress = st.progress(0)
    status = st.empty()
    eta = st.empty()

    class_counts = {i: 0 for i in range(len(COCO_CLASSES))}

    total = len(uploaded_files)
    start = time.time()

    if show_previews:
        st.subheader("Annotated Previews")

    for i, file in enumerate(uploaded_files, start=1):
        status.text(f"Processing {i}/{total}: {file.name}")

        # Read bytes and immediately discard the file reference
        img_bytes = file.read()
        preview, classes = run_detection(interpreter, img_bytes, file.name, conf_thresh)
        del img_bytes  # free upload bytes right away

        for c in classes:
            class_counts[c] += 1

        # Render preview immediately (don't accumulate in a list)
        if show_previews and preview is not None:
            st.markdown(f"**{file.name}**")
            st.image(preview, use_container_width=True)

        if preview is not None:
            del preview

        elapsed = time.time() - start
        avg = elapsed / i
        remaining = avg * (total - i)
        progress.progress(i / total)
        eta.text(f"Estimated time remaining: {remaining:.1f}s")

        # Force garbage collection every BATCH_SIZE images
        if i % BATCH_SIZE == 0:
            gc.collect()

    # Final GC sweep
    gc.collect()

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

    shutil.make_archive("dataset_export", "zip", "dataset")
    with open("dataset_export.zip", "rb") as f:
        st.download_button(
            "Download Dataset ZIP",
            f,
            file_name="dataset.zip",
            mime="application/zip"
        )

    st.success("Dataset exported in YOLO format ✅")
