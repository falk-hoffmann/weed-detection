import streamlit as st


st.set_page_config(
    page_title="About the Project",
    page_icon="🌱",
    layout="wide",
)


# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(128, 128, 128, 0.035);
        border-radius: 10px;
    }

    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.55;
        margin-bottom: 0.2rem;
    }

    .model-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .model-subtitle {
        opacity: 0.65;
        margin-bottom: 1rem;
    }

    .pipeline {
        text-align: center;
        font-size: 1.05rem;
        padding: 0.8rem;
        border-radius: 8px;
        background-color: rgba(128, 128, 128, 0.06);
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.title("Precision Farming · Weed Detection")

st.markdown(
    """
    Development and evaluation of a computer-vision pipeline for
    **maize and weed detection under varying field conditions**.

    The project compares two real-time object-detection architectures,
    evaluates training and augmentation strategies, and benchmarks
    inference performance across PyTorch and ONNX backends.
    """
)

st.markdown(
    """
    <div class="pipeline">
        CornWeed Dataset
        &nbsp; → &nbsp;
        Training & Augmentation
        &nbsp; → &nbsp;
        YOLO26 / RF-DETR
        &nbsp; → &nbsp;
        Evaluation
        &nbsp; → &nbsp;
        Inference Benchmark      
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------

st.markdown("## Detection Models")

yolo_col, detr_col = st.columns(2, gap="large")


with yolo_col:

    with st.container(border=True):

        st.markdown(
            """
            <div class="section-label">CNN-based detector</div>
            <div class="model-title">YOLO26</div>
            <div class="model-subtitle">
                Ultralytics · Real-time object detection
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            **Evaluated model sizes**

            `Nano (n)` · `Small (s)` · `Medium (m)`

            **Training**

            - pretrained weights
            - fine-tuned on CornWeed
            - bounding-box detection
            - two target classes: **Weeds** and **Maize**
            - controlled augmentation experiments
            - best epoch selected using **mAP50-95**

            **Evaluated augmentations**

            - Mosaic
            - vertical flipping
            - HSV value / illumination variation

            **Inference**

            - native PyTorch (`.pt`)
            - exported ONNX (`.onnx`)
            - confidence-threshold controlled inference
            """
        )


with detr_col:

    with st.container(border=True):

        st.markdown(
            """
            <div class="section-label">Transformer-based detector</div>
            <div class="model-title">RF-DETR</div>
            <div class="model-subtitle">
                Roboflow · Real-Time Detection Transformer
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            **Evaluated model sizes**

            `Nano (n)` · `Small (s)` · `Medium (m)`

            **Architecture**

            - transformer-based object detector
            - DINOv2 vision-transformer backbone
            - pretrained weights
            - fine-tuned on CornWeed
            - bounding-box detection
            - two target classes: **Weeds** and **Maize**

            **Training**

            - native RF-DETR training pipeline
            - default resize / crop scale jitter
            - horizontal flipping
            - exponential moving average (EMA)
            - best epoch evaluated using **mAP50-95**

            **Inference**

            - native PyTorch (`.pth`)
            - exported ONNX (`.onnx`)
            - ONNX Runtime inference
            """
        )


st.info(
    "The comparison focuses on the accuracy–latency trade-off rather "
    "than accuracy alone. Model size, framework and inference backend "
    "are therefore evaluated separately."
)

st.divider()


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------

st.markdown("## Dataset · CornWeed")

dataset_info, dataset_split = st.columns([1.3, 1], gap="large")


with dataset_info:

    with st.container(border=True):

        st.markdown(
            """
            ### Dataset characteristics

            The **CornWeed Dataset** contains agricultural field images
            recorded for maize and weed object detection.

            - **3,574 labelled images**
            - **4,981 additional unlabelled images**
            - bounding-box annotations
            - YOLO and COCO annotation formats
            - classes used in this project:
              **Weeds** and **Maize**
            - image resolutions:
              **1280 × 720** and **640 × 480**
            - real agricultural field conditions
            """
        )


with dataset_split:

    with st.container(border=True):

        st.markdown(
            """
            ### Project split

            **Training**

            `2,862 images`

            **Validation**

            `712 images`

            **Unlabelled data**

            `4,981 images`

            The predefined train/validation split was retained to keep
            experiments reproducible and directly comparable.
            """
        )


st.divider()


# -------------------------------------------------------------------
# Experimental design
# -------------------------------------------------------------------

st.markdown("## Experimental Design")

exp1, exp2, exp3 = st.columns(3, gap="medium")


with exp1:

    with st.container(border=True):

        st.markdown(
            """
            ### 1 · Training

            Both detector families were fine-tuned on the same
            agricultural detection task.

            Model variants:

            - Nano
            - Small
            - Medium

            Primary evaluation metric:

            **mAP50-95**
            """
        )


with exp2:

    with st.container(border=True):

        st.markdown(
            """
            ### 2 · Robustness

            YOLO augmentation experiments investigated the influence of:

            - Mosaic
            - vertical flipping
            - illumination variation (`hsv_v`)

            This specifically addresses challenging field conditions
            such as shadows and changing sunlight.
            """
        )


with exp3:

    with st.container(border=True):

        st.markdown(
            """
            ### 3 · Inference

            Deployment performance was benchmarked using:

            - PyTorch
            - ONNX
            - fixed validation images
            - identical image set per model
            - warm-up runs
            - per-image latency measurement

            Reported statistics include mean, median and P95 latency.
            """
        )


st.divider()


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

st.markdown("## Evaluation Strategy")

eval_col, benchmark_col = st.columns(2, gap="large")


with eval_col:

    with st.container(border=True):

        st.markdown(
            """
            ### Detection quality

            Models are compared using:

            - **Precision**
            - **Recall**
            - **mAP50**
            - **mAP50-95**

            The best validation epoch of each training run is retained
            for comparison.

            Error analysis additionally considers typical agricultural
            failure cases such as small weeds, occlusion, shadows,
            soil confusion and crop/weed confusion.
            """
        )


with benchmark_col:

    with st.container(border=True):

        st.markdown(
            """
            ### Accuracy vs. latency

            Inference benchmarking uses a fixed sample of validation
            images for all model/backend combinations.

            Compared dimensions:

            - YOLO26 vs. RF-DETR
            - Nano vs. Small vs. Medium
            - PyTorch vs. ONNX

            The resulting latency–accuracy plot visualizes the practical
            trade-off between detection quality and inference speed.
            """
        )


st.divider()


# -------------------------------------------------------------------
# Deployment
# -------------------------------------------------------------------

st.markdown("## Deployment Pipeline")





deploy2, deploy3 = st.columns([4.5,3], gap="medium")
with deploy2:

    with st.container(border=True):

        st.markdown(
            """
            ### Training

            **Initial experiments · Google Colab**

            - first manual training runs and model experiments
            - NVIDIA T4 / L4 GPUs
            - interactive development and validation of the training pipeline
            - model checkpoints persisted via Google Drive

            **Automated training · Google Cloud Batch**

            - GPU-based training jobs
            - reproducible containerized environment
            - dataset and artifacts stored in Google Cloud Storage
            - suitable for finite, compute-intensive training workloads
            - automated execution of larger experiment grids

            The workflow evolved from interactive Colab experiments
            to reproducible cloud-based batch training as the number
            of experiments increased.
            """
        )


with deploy3:

    with st.container(border=True):
    
            st.markdown(
                """
                ### Storage
    
                **Google Cloud Storage**
    
                Used for datasets, training artifacts and model checkpoints.
    
                Training artifacts are additionally persisted for later
                evaluation and deployment.
                """
            )

    with st.container(border=True):

        st.markdown(
            """
            ### Application

            **Streamlit**

            Interactive inference and model comparison.

            Target deployment:

            **Docker + Google Cloud Run**
            """
        )


st.divider()


# -------------------------------------------------------------------
# Limitations and future work
# -------------------------------------------------------------------

st.markdown("## Limitations & Future Work")

limitations, future = st.columns(2, gap="large")


with limitations:

    with st.container(border=True):

        st.markdown(
            """
            ### Current limitations

            Detection performance can be affected by:

            - strong shadows and changing illumination
            - very small weeds
            - occlusion by maize plants
            - visually complex soil structures
            - conditions not represented in the training data

            **Validation strategy**

            The predefined train/validation split was retained.
            K-fold cross-validation was not performed due to project
            scope and computational cost.

            Random image-level splitting could also introduce leakage
            because consecutive field images may be visually similar.
            """
        )


with future:

    with st.container(border=True):

        st.markdown(
            """
            ### Future work

            **Model validation**

            - group-based cross-validation by field sequence,
              recording session or environmental condition
            - evaluation on additional fields and growth stages

            **Optimization**

            - TensorRT / FP16 inference on NVIDIA hardware
            - real-time video inference
            - edge-device benchmarking

            **Precision weeding**

            - stem / plant-center localization
            - keypoint detection
            - spatial safety zones around maize
            - integration with a targeted actuation system
            """
        )


st.divider()


# -------------------------------------------------------------------
# Sources and licenses
# -------------------------------------------------------------------

st.markdown("## Sources & Licenses")

source1, source2, source3 = st.columns(3, gap="medium")


with source1:

    with st.container(border=True):

        st.markdown(
            """
            ### Ultralytics YOLO26

            **Provider:** Ultralytics

            **License:** AGPL-3.0 or Ultralytics Enterprise License

            YOLO26 is used as the convolutional real-time object
            detection model family in this project.
            """
        )

        st.link_button(
            "YOLO26 documentation",
            "https://docs.ultralytics.com/models/yolo26/",
            use_container_width=True,
        )

        st.link_button(
            "Ultralytics licensing",
            "https://www.ultralytics.com/license",
            use_container_width=True,
        )


with source2:

    with st.container(border=True):

        st.markdown(
            """
            ### RF-DETR

            **Provider:** Roboflow

            **License:** Apache License 2.0 for the core package and
            Nano–Large model family used in this project.

            RF-DETR is the transformer-based comparison architecture.
            """
        )

        st.link_button(
            "RF-DETR documentation",
            "https://rfdetr.roboflow.com/",
            use_container_width=True,
        )

        st.link_button(
            "RF-DETR repository",
            "https://github.com/roboflow/rf-detr",
            use_container_width=True,
        )


with source3:

    with st.container(border=True):

        st.markdown(
            """
            ### CornWeed Dataset

            **Authors:** Iqbal et al.

            **License:** CC BY 4.0

            **DOI:** `10.5281/zenodo.7961764`

            Dataset for maize and weed object detection in agricultural
            field environments.
            """
        )

        st.link_button(
            "CornWeed · Zenodo",
            "https://doi.org/10.5281/zenodo.7961764",
            use_container_width=True,
        )

        st.link_button(
            "CC BY 4.0",
            "https://creativecommons.org/licenses/by/4.0/",
            use_container_width=True,
        )


st.caption(
    "CornWeed: Iqbal, N., Manss, C., Scholz, C., Koenig, D., "
    "Igelbrink, M. & Ruckelshausen, A. (2023). "
    "CornWeed Dataset: A dataset for training maize and weed object "
    "detectors for agricultural machines."
)