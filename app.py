import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import zipfile
import os
import io
import shutil
import requests

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
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


# ---------------- NMS HELPERS ----------------
def compute_iou(box1, boxes):
    x1, y1, x2, y2 = box1
    xx1 = np.maximum(x1, boxes[:, 0])
    yy1 = np.maximum(y1, boxes[:, 1])
    xx2 = np.minimum(x2, boxes[:, 2])
    yy2 = np.minimum(y2, boxes[:, 3])

    inter_w = np.maximum(0, xx2 - xx1)
    inter_h = np.maximum(0, yy2 - yy1)
    inter_area = inter_w * inter_h

    box_area = (x2 - x1) * (y2 - y1)
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union_area = box_area + boxes_area - inter_area
    return inter_area / (union_area + 1e-6)


def non_max_suppression(dets, iou_threshold=0.5):
    if not dets:
        return []
    boxes = np.array([d["bbox"] for d in dets])
    scores = np.array([d["score"] for d in dets])
    clss = np.array([d["class_id"] for d in dets])

    keep = []
    for c in np.unique(clss):
        mask = clss == c
        b = boxes[mask]
        s = scores[mask]
        idxs = np.argsort(-s)

        while len(idxs):
            best = idxs[0]
            keep.append(np.where(mask)[0][best])
            if len(idxs) == 1:
                break

            ious = compute_iou(b[best], b[idxs[1:]])
            idxs = idxs[1:][ious < iou_threshold]

    return [dets[i] for i in keep]


# ---------------- DETECTION + DATASET EXPORT ----------------
def run_detection(img_bytes, conf_thresh=0.7, output_dataset="dataset"):
    download_model()

    # Load image
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (640, 640))
    h, w, _ = img_resized.shape

    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    inp = rgb.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, 0)

    # Load TFLite
    interpreter = tf.lite.Interpreter(MODEL_PATH)
    interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    interpreter.set_tensor(in_details[0]["index"], inp)
    interpreter.invoke()

    output = interpreter.get_tensor(out_details[0]["index"])[0]

    # Parse detections
    detections = []
    for i, det in enumerate(output):
        cx, cy, bw, bh = det[:4]
        conf = det[4]
        probs = det[5:]
        cls_id = int(np.argmax(probs))
        score = conf * probs[cls_id]

        if score < conf_thresh:
            continue

        cx *= w
        cy *= h
        bw *= w
        bh *= h

        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class_id": cls_id,
            "score": score,
            "index": i
        })

    detections = non_max_suppression(detections)

    # Build dataset folder
    img_dir = os.path.join(output_dataset, "images")
    lbl_dir = os.path.join(output_dataset, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    saved_files = []

    # Save crops + YOLO labels
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = img_resized[max(0,y1):y2, max(0,x1):x2]

        crop_name = f"crop_{det['index']}.jpg"
        crop_path = os.path.join(img_dir, crop_name)
        cv2.imwrite(crop_path, crop)

        # YOLO normalized labels
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        label_name = f"crop_{det['index']}.txt"
        label_path = os.path.join(lbl_dir, label_name)

        with open(label_path, "w") as f:
            f.write(f"{det['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        saved_files.append((crop_path, label_path))

    # Draw annotated preview
    preview = img_resized.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0,255,0), 2)

    return preview, saved_files


# ---------------- STREAMLIT UI ----------------
st.title("📦 YOLO Dataset Builder from Single Image")

uploaded = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
conf_thresh = st.slider("Confidence Threshold", 0.1, 0.95, 0.7)

if uploaded:
    st.image(uploaded, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Detection & Build Dataset"):
        preview, saved_files = run_detection(uploaded.read(), conf_thresh)

        st.subheader("Annotated Preview")
        st.image(preview, use_container_width=True)

        st.success(f"{len(saved_files)} crops saved to dataset/")

        # Create downloadable ZIP
        zip_path = "dataset_export.zip"
        shutil.make_archive("dataset_export", "zip", "dataset")

        with open(zip_path, "rb") as f:
            st.download_button(
                "Download Dataset ZIP",
                f,
                file_name="dataset.zip",
                mime="application/zip"
            )

        st.info("Dataset exported in YOLO format (images + labels).")
