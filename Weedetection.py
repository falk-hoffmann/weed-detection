import os
from pathlib import Path
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import cast
import random as rn
import plotly.express as px
import pandas as pd
from PIL import Image, ImageDraw
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
import time
import supervision as sv
import onnxruntime as ort
import numpy as np
import torchvision.transforms.functional as F
import plotly.graph_objects as go



working_path = Path.cwd()
content_path = working_path / "content"



@st.cache_resource
def load_yolo(checkpoint_path):
    # Load YOLO model
    return YOLO(checkpoint_path)

@st.cache_resource
def load_rfdetr(model_size, checkpoint_path):
    # Map model size to RF-DETR architecture
    model_classes = {
        "n": RFDETRNano,
        "s": RFDETRSmall,
        "m": RFDETRMedium,
    }

    return model_classes[model_size](
        pretrain_weights=str(checkpoint_path)
    )

@st.cache_resource
def load_rfdetr_onnx(checkpoint_path):
    # Load exported RF-DETR ONNX model
    return ort.InferenceSession(
        str(checkpoint_path),
        providers=["CPUExecutionProvider"],
    )
st.markdown("""
<style>

/* Top control cards */
div[data-testid="stHorizontalBlock"]:has(
    div[data-testid="stSlider"]
) > div[data-testid="stColumn"] {
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 8px;
    padding: 0.8rem 1rem;
}

/* Metric cards */
div[data-testid="stMetric"] {
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 8px;
    padding: 0.8rem 1rem;
}



<style>
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(128, 228, 128, 0.04);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


st.set_page_config(
    page_title="Detection",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 Weed Detection")
st.subheader("... with fine-tuned YOLO26 and RF-DETR")
st.space("medium")
st.subheader("Prediction", icon="🔍")


col1_0, col1_1, col1_2, col1_3 = st.columns(4)
with col1_0:
    st.caption("Model framework")
    framework_toggle = st.toggle(
        "YOLO26 / RF-DETR",
        value=False,
    )
with col1_1:
    model_size = st.radio(
        "Model size",
        options=["n", "s", "m"],
        #captions=["nano", "small", "medium"],
        horizontal=True,
    )
with col1_2:
    # Select inference backend
    st.caption("Inference backend")

    onnx_toggle = st.toggle(
        "PyTorch / ONNX",
        value=False,
        )
with col1_3:
    # Confidence Threshold
    conf = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
    )

st.space()


framework = "detr" if framework_toggle else "yolo"
backend = "onnx" if onnx_toggle else "torch"

if framework == "detr":
    
    if backend == "torch":

        suffix = ".pth"
        
        model_checkpoint_path = Path(working_path / "maize_dataset" / "runs" / "weights" / f"{framework}_{model_size}{suffix}" )

        model = load_rfdetr(
            model_size,
            model_checkpoint_path,
        )

    else:

        suffix = ".onnx"
        
        model_checkpoint_path = Path(working_path / "maize_dataset" / "runs" / "weights" / f"{framework}_{model_size}{suffix}" )

        model = load_rfdetr_onnx(
            model_checkpoint_path,
        )

elif framework == "yolo":

    suffix = ".onnx" if onnx_toggle else ".pt"

    model_checkpoint_path = Path(working_path / "maize_dataset" / "runs" / "weights" / f"{framework}_{model_size}{suffix}" )

    model = load_yolo(model_checkpoint_path)


# read metrics csv for plotting
df = pd.read_csv( Path(working_path / "maize_dataset"  / "runs" / f"metrics_{framework}_{model_size}.csv") )



image = []
annotated_image = []

def new_random_image():
    st.session_state.valfile = rn.choice(valfiles)

with open(content_path / "maize_dataset" / "valid_sorted.txt", "r") as f:
    valfiles_raw = f.read().splitlines()

valfiles = []
for line in valfiles_raw:
    valfiles.append(Path(content_path / "maize_dataset" / line))


# Defaults beim ersten Start
if "valfile" not in st.session_state:
    st.session_state.valfile = rn.choice(valfiles)

# open image and convert to RGB
image = Image.open(st.session_state.valfile).convert("RGB")
is_validation_image = True



class_names = {
    0: "Weeds",
    1: "Maize",
}

class_colors = {
    0: "red",
    1: "blue",
}

def predict_rfdetr_onnx(session, image, threshold):
    # Store original image dimensions
    img_w, img_h = image.size

    # Get ONNX input dimensions
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_h, input_w = input_info.shape[2:4]

    # Preprocess image
    image_tensor = F.to_tensor(image)
    image_tensor = F.resize(
        image_tensor,
        [input_h, input_w],
        antialias=False,
    )

    image_tensor = F.normalize(
        image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    image_array = image_tensor.unsqueeze(0).numpy().astype(np.float32)

    # Run ONNX inference
    outputs = session.run(
        None,
        {input_name: image_array},
    )

    # Identify outputs by name
    output_names = [output.name for output in session.get_outputs()]

    boxes_idx = next(
        i for i, name in enumerate(output_names)
        if "dets" in name
    )

    logits_idx = next(
        i for i, name in enumerate(output_names)
        if "labels" in name
    )

    boxes = outputs[boxes_idx][0]
    logits = outputs[logits_idx][0]

    # Convert logits to probabilities
    scores = 1.0 / (1.0 + np.exp(-logits))

    # Remove background class
    scores = scores[:, :-1]

    class_ids = scores.argmax(axis=1)
    confidences = scores.max(axis=1)

    # Apply confidence threshold
    keep = confidences >= threshold

    boxes = boxes[keep]
    class_ids = class_ids[keep]

    prediction_boxes = []

    # Convert normalized cxcywh boxes to image-space xyxy
    for box, cls in zip(boxes, class_ids):
        cx, cy, w, h = box

        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h

        prediction_boxes.append(
            [x1, y1, x2, y2, int(cls)]
        )

    return prediction_boxes


def draw_boxes(image, boxes):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for box in boxes:
        x1, y1, x2, y2, cls = box

        class_id = int(cls)
        color = class_colors[class_id]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=3,
        )

        draw.text(
            (x1, y1),
            class_names[class_id],
            fill=color,
            )

    return annotated


prediction_boxes = []
inference_time = None


if framework == "yolo":
    start_time = time.perf_counter()

    result = cast(
        list[Results],
        model.predict(
            source=image,
            rect=True,
            conf=conf,
            save=False,
            show_labels=False,
            show_conf=False,
            show_boxes=True,
            verbose=True,
        )
    )[0]

    inference_time = (time.perf_counter() - start_time) * 1000

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])

            prediction_boxes.append(
                [x1, y1, x2, y2, cls]
            )

elif framework == "detr":
    start_time = time.perf_counter()

    if backend == "torch":
        # Run RF-DETR with PyTorch
        detections = cast(
            sv.Detections,
            model.predict(
                image,
                threshold=conf,
            )
        )

        if detections.class_id is not None:
            for xyxy, cls in zip(
                detections.xyxy,
                detections.class_id,
            ):
                x1, y1, x2, y2 = xyxy

                prediction_boxes.append(
                    [x1, y1, x2, y2, int(cls)]
                )

    else:
        # Run RF-DETR with ONNX Runtime
        prediction_boxes = predict_rfdetr_onnx(
            model,
            image,
            conf,
        )

    inference_time = (time.perf_counter() - start_time) * 1000


annotated_image = draw_boxes(
    image,
    prediction_boxes,
)

ground_truth_image = None

if is_validation_image:
    image_path = st.session_state.valfile

    label_path = Path(
        str(image_path).replace("/images/", "/labels/")
    ).with_suffix(".txt")

    ground_truth_boxes = []

    if label_path.exists():
        img_w, img_h = image.size

        with open(label_path, "r") as f:
            for line in f:
                cls, x, y, w, h = map(float, line.split()[:5])

                x1 = (x - w / 2) * img_w
                y1 = (y - h / 2) * img_h
                x2 = (x + w / 2) * img_w
                y2 = (y + h / 2) * img_h

                ground_truth_boxes.append(
                    [x1, y1, x2, y2, int(cls)]
                )

    ground_truth_image = draw_boxes(
        image=image,
        boxes=ground_truth_boxes,
    )

   


if is_validation_image and ground_truth_image is not None:
    col_pred, col_gt = st.columns(2)

    with col_pred:
        st.markdown("**Model prediction**")
        st.image(
            annotated_image,
            width="content",
        )

    with col_gt:
        st.markdown("**Ground truth**")
        st.image(
            ground_truth_image,
            width="content",
            )

else:
    st.image(
        annotated_image,
        width="content",
    )


col_time, col_fps , col_next = st.columns([50,40,10])
with col_next:
    new_random_val = st.button(
        type="primary",
        width="content",
        icon="🎲", 
        label="Next image",
        on_click=new_random_image,
    )

if inference_time is not None:
    col_time.metric(
        "Inference time",
        f"{inference_time:.1f} ms"
        )

    # Show inference time below img:
    fps = 1000 / inference_time

    col_fps.metric(
        "Theoretical FPS",
        f"{fps:.1f}"
        )


    


################
# plot metrics
################

metric_columns = {
    "yolo": {
        "metrics/mAP50-95(B)": "mAP50-95",
        "metrics/mAP50(B)": "mAP50",
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
    },
    "detr": {
        "val/ema_mAP_50_95": "mAP50-95",
        "val/ema_mAP_50": "mAP50",
        "val/precision": "precision",
        "val/recall": "recall",
    },
}
df = df.rename(columns=metric_columns[framework])


# Ultralytics-Spalten enthalten teilweise führende Leerzeichen
df.columns = df.columns.str.strip()

st.space("medium")
st.subheader("Model Performance", icon= "📊")
col1, col2, col3, col4 = st.columns(4)

best_idx = df["mAP50-95"].idxmax()
best_row = df.loc[best_idx]

col1.metric("mAP50-95", f"{best_row['mAP50-95']:.3f}")
col2.metric("mAP50", f"{best_row['mAP50']:.3f}")
col3.metric("Precision", f"{best_row['precision']:.3f}")
col4.metric("Recall", f"{best_row['recall']:.3f}")



chart_data = df.set_index("epoch")[
    ["mAP50-95", "mAP50", "precision", "recall"]
]
