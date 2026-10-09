# Precision Farming · Weed Detection

Computer-vision pipeline for detecting **maize and weeds under realistic agricultural field conditions**.

The project compares CNN- and transformer-based object detectors, investigates robustness to challenging field conditions, benchmarks inference performance, and provides an interactive Streamlit application for exploring the results.

## Project Scope

The long-term motivation is autonomous precision weeding:

**Field Image → Detect Crop & Weed → Localize Target → Verify Safety → Actuate**

This project focuses specifically on the **object-detection stage**:

- Bounding-box detection
- Two classes: **Weeds** and **Maize**
- YOLO26 and RF-DETR
- Nano, Small, and Medium model variants
- Accuracy, robustness, and inference-speed evaluation

Target localization, safety zones, and physical actuation are outside the current project scope.

## Dataset

The project uses the **CornWeed Dataset** by Iqbal et al.

| Split | Images |
|---|---:|
| Training | 2,862 |
| Validation | 712 |
| Unlabelled | 4,981 |

The dataset contains agricultural field images with bounding-box annotations for maize and weeds. Images include realistic variations in illumination, shadows, plant size, occlusion, and background structure.

The predefined train/validation split is retained for reproducibility and direct comparison between experiments.

## Detection Models

### YOLO26

Ultralytics YOLO26 is used as the CNN-based real-time object detector.

Evaluated variants:

- Nano
- Small
- Medium

Training includes pretrained weights, fine-tuning on CornWeed, and controlled augmentation experiments.

### RF-DETR

RF-DETR is used as the transformer-based comparison architecture.

Evaluated variants:

- Nano
- Small
- Medium

The models are fine-tuned on the same maize/weed detection task to enable comparison with YOLO26.

## Experiments

The project investigates three main areas.

### 1. Detection Quality

Models are evaluated using:

- Precision
- Recall
- mAP50
- mAP50-95

The primary comparison metric is **mAP50-95**.

### 2. Robustness

YOLO augmentation experiments investigate:

- Mosaic
- Vertical flipping
- HSV value / illumination variation

The goal is to evaluate robustness under challenging agricultural conditions such as shadows, changing sunlight, small weeds, and complex soil backgrounds.

### 3. Inference Performance

Inference benchmarking compares:

- YOLO26 vs. RF-DETR
- Nano vs. Small vs. Medium
- PyTorch vs. ONNX
- Detection quality vs. inference latency

All model/backend combinations are benchmarked using a fixed set of validation images with warm-up runs and per-image latency measurements.

Reported latency statistics include mean, median, and P95.

## Application

An interactive **Streamlit** application provides:

- Maize and weed detection
- Prediction vs. ground-truth visualization
- Adjustable confidence threshold
- Model-size selection
- Model-quality metrics
- Training experiment results
- Inference benchmarks
- Accuracy–latency comparison

For the public deployment, inference is intentionally simplified to the YOLO Nano, Small, and Medium PyTorch models.

## Training & Deployment

The project evolved from interactive experimentation to a reproducible cloud workflow:

```text
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

Initial experiments were developed interactively in Google Colab. As the experiment grid increased, training was moved to containerized Google Cloud Batch jobs.

The public Streamlit application is packaged as a Docker container and deployed using Google Cloud Run.

## Project Structure

```text
.
├── app.py
├── pages/
│   ├── background.py
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

## Limitations

Current limitations include:

- Strong shadows and changing illumination
- Very small weeds
- Occlusion by maize plants
- Complex soil structures
- Generalization to conditions not represented in the training data

The predefined train/validation split is retained. K-fold cross-validation was not performed due to project scope and computational cost.

## Future Work

Potential extensions include:

- Group-based validation by field sequence or environmental condition
- Evaluation on additional fields and growth stages
- Real-time video inference
- Edge-device benchmarking
- TensorRT / FP16 optimization
- Stem or plant-center localization
- Keypoint detection
- Spatial safety zones around maize
- Integration with a targeted actuation system

## Dataset & Licenses

**CornWeed Dataset**  
Iqbal et al. (2023)  
DOI: `10.5281/zenodo.7961764`  
License: CC BY 4.0

**Ultralytics YOLO26**  
License: AGPL-3.0 or Ultralytics Enterprise License

**RF-DETR**  
The Nano–Large model family used in this project is provided under the Apache License 2.0.

## Tech Stack

`Python` · `PyTorch` · `Ultralytics` · `RF-DETR` · `Pandas` · `Plotly` · `Streamlit` · `Docker` · `Google Cloud Batch` · `Google Cloud Storage` · `Google Cloud Run`
