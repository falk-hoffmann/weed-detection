# 🌱 Precision Farming · Weed Detection

Computer-vision pipeline for detecting **maize and weeds under realistic
agricultural field conditions**.


The project compares CNN- and transformer-based object detectors,
investigates robustness to challenging field conditions, benchmarks
inference performance, and provides an interactive
[Streamlit](https://streamlit.io/) application for exploring the
results.


🕹️ **Public demo:**
[weed-detection.cloud.run](https://weed-detection.cloud.run/) –– a
lightweight deployment using **PyTorch with YOLO26 Nano, Small, and
Medium only**.


------------------------------------------------------------------------

## 🎯 Project Scope

The long-term motivation is autonomous precision weeding:

> **Field Image → Detect Crop & Weed → Localize Target → Verify Safety →
> Actuate**

This project focuses specifically on the **object-detection stage**:

-   Bounding-box detection
-   Two classes: **Weeds** and **Maize**
-   YOLO26 and RF-DETR
-   Nano, Small, and Medium model variants
-   Accuracy, robustness, and inference-speed evaluation

**Target localization, safety zones, and physical actuation are outside
the current project scope.**

------------------------------------------------------------------------

## 🌽 Dataset

The project uses the **CornWeed Dataset** by Iqbal et al. (2023),
publicly available through
[Zenodo](https://doi.org/10.5281/zenodo.7961764).

  Split          Images
  ------------ --------
  Training        2,862
  Validation        712
  Unlabelled      4,981

The dataset contains agricultural field images with bounding-box
annotations for maize and weeds. Images include realistic variations in
illumination, shadows, plant size, occlusion, and background structure.

The predefined train/validation split is retained for **reproducibility
and direct comparison between experiments**.

The full validation split contains **712 images**. To keep the
interactive prediction test and inference benchmark lightweight and
reproducible, only a **fixed random sample of 100 validation images** is
used there rather than all 712 validation images.

------------------------------------------------------------------------

## 🧠 Detection Models

### YOLO26

[Ultralytics YOLO](https://docs.ultralytics.com/) is used as the
CNN-based real-time object detector.

**Evaluated variants:**

-   Nano
-   Small
-   Medium

Training includes pretrained weights, fine-tuning on CornWeed, and
controlled augmentation experiments.

### RF-DETR

[RF-DETR](https://github.com/roboflow/rf-detr) is used as the
transformer-based comparison architecture.

**Evaluated variants:**

-   Nano
-   Small
-   Medium

The models are fine-tuned on the same maize/weed detection task to
enable a direct comparison with YOLO26.

------------------------------------------------------------------------

## 🔬 Experiments

The experimental design focuses on three questions:

  ---------------------------------------------------------------------
  Area                               Main Question
  ---------------------------------- ----------------------------------
  **Detection Quality**              How accurately are maize and weeds
                                     detected?

  **Robustness**                     How well do models handle
                                     difficult field conditions?

  **Inference Performance**          What is the trade-off between
                                     accuracy and speed?
  ---------------------------------------------------------------------

### 1. Detection Quality

Models are evaluated using:

-   Precision
-   Recall
-   mAP50
-   **mAP50-95**

The primary comparison metric is **mAP50-95**, providing a stricter
assessment across multiple IoU thresholds.

### 2. Robustness

Controlled YOLO augmentation experiments investigate:

-   Mosaic
-   Vertical flipping
-   HSV value / illumination variation

The goal is to evaluate robustness under challenging agricultural
conditions such as **shadows, changing sunlight, small weeds, occlusion,
and complex soil backgrounds**.

### 3. Inference Performance

Inference benchmarking compares:

-   YOLO26 vs. RF-DETR
-   Nano vs. Small vs. Medium
-   PyTorch vs. ONNX
-   Detection quality vs. inference latency

All model/backend combinations are benchmarked using a **fixed random
sample of 100 images from the 712-image validation split**, including
warm-up runs and per-image latency measurements.

Reported latency statistics include:

`Mean` · `Median` · `P95`

This enables a direct evaluation of the **accuracy--latency trade-off**
between model architectures, sizes, and inference backends.

------------------------------------------------------------------------

## 🖥️ Application

An interactive [Streamlit](https://streamlit.io/) application provides:

-   Maize and weed detection
-   Prediction vs. ground-truth visualization
-   Adjustable confidence threshold
-   Model-size selection
-   Model-quality metrics
-   Training experiment results
-   Inference benchmarks
-   Accuracy--latency comparison

The full **GitHub version** contains the experimental application with
**YOLO26 and RF-DETR** models and supports both **PyTorch and ONNX**
backends.

The lightweight **public Streamlit deployment** is available at
**[weed-detection.cloud.run](https://weed-detection.cloud.run/)**. To
keep the cloud deployment smaller and more resource-efficient, this
version is intentionally limited to **PyTorch with YOLO26 Nano, Small,
and Medium**.

The prediction test uses a **fixed random sample of 100 validation
images** instead of all **712 validation images** in the CornWeed
validation split.

------------------------------------------------------------------------

## ☁️ Training & Deployment

The project evolved from interactive experimentation to a reproducible
cloud workflow:

``` text
Initial experiments
        ↓
Google Colab · NVIDIA T4 / L4
        ↓
Google Cloud Batch · GPU training
        ↓
Google Cloud Storage
        ↓
Model evaluation & benchmarking
        ↓
Streamlit
        ↓
Docker
        ↓
Google Cloud Run
```

Initial experiments were developed interactively in [Google
Colab](https://colab.research.google.com/), using NVIDIA T4 and L4 GPUs.

As the experiment grid increased, training was moved to containerized
[Google Cloud Batch](https://cloud.google.com/batch) jobs, with
experiment artifacts stored in [Google Cloud
Storage](https://cloud.google.com/storage).

The public Streamlit application is packaged as a
[Docker](https://www.docker.com/) container and deployed as a serverless
service using [Google Cloud Run](https://cloud.google.com/run).

This separates the computationally expensive **training workflow** from
the lightweight **public inference application**.

------------------------------------------------------------------------

## 📁 Project Structure

``` text
.
├── app.py
├── pages/
│   ├── background.py
│   ├── prediction.py
│   ├── model_quality.py
│   ├── speed_benchmark.py
│   ├── experiments.py
│   └── about.py
├── maize_dataset/
│   └── runs/
│       └── weights/
│           ├── yolo_n.pt
│           ├── yolo_s.pt
│           └── yolo_m.pt
├── requirements.txt
├── Dockerfile
└── README.md
```

------------------------------------------------------------------------

## ⚠️ Limitations

Current limitations include:

-   Strong shadows and changing illumination
-   Very small weeds
-   Occlusion by maize plants
-   Complex soil structures
-   Generalization to conditions not represented in the training data

The predefined train/validation split is retained. **K-fold
cross-validation was not performed** due to project scope and
computational cost.

The current system detects plants using bounding boxes. It does **not**
determine a precise physical treatment point or perform autonomous
actuation.

------------------------------------------------------------------------

## 🔭 Future Work

Potential extensions include:

-   Group-based validation by field sequence or environmental condition
-   Evaluation on additional fields and growth stages
-   Real-time video inference
-   Edge-device benchmarking
-   TensorRT / FP16 optimization
-   Stem or plant-center localization
-   Keypoint detection
-   Spatial safety zones around maize
-   Integration with a targeted actuation system

A future precision-weeding system could extend the current detector
from:

> **Where is the weed?**

to:

> **Where can the weed be treated safely?**

This would require more precise localization, crop safety constraints,
and integration with the physical actuator.

------------------------------------------------------------------------

## 📜 Dataset & Licenses

### CornWeed Dataset

**Iqbal et al. (2023)**\
[DOI: 10.5281/zenodo.7961764](https://doi.org/10.5281/zenodo.7961764)\
License: **CC BY 4.0**

### Ultralytics YOLO26

Developed by [Ultralytics](https://www.ultralytics.com/).\
License: **AGPL-3.0 or Ultralytics Enterprise License**

See the [Ultralytics licensing
information](https://www.ultralytics.com/license) for details.

### RF-DETR

Developed by [Roboflow](https://github.com/roboflow/rf-detr).\
The Nano--Large model family used in this project is provided under the
**Apache License 2.0**.

------------------------------------------------------------------------

## 🛠️ Tech Stack

**Machine Learning**

`Python` · `PyTorch` · `Ultralytics` · `RF-DETR`

**Data & Visualization**

`Pandas` · `Plotly`

**Application**

`Streamlit`

**Cloud & Deployment**

`Docker` · `Google Cloud Batch` · `Google Cloud Storage` ·
`Google Cloud Run`

------------------------------------------------------------------------

### Project focus

**Robust weed detection under realistic field conditions --- from model
training and evaluation to a deployable computer-vision application.**
