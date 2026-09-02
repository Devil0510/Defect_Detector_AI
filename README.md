# AI-Based Steel Surface Defect Severity Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8%20%2F%2011-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

A research-grade, reproducible, modular computer-vision and machine-learning pipeline for steel surface defect detection, quantitative feature extraction, deep neural severity estimation, Monte Carlo Dropout epistemic uncertainty quantification, and spatial Grad-CAM severity mapping using the NEU-DET (Northeastern University Surface Defect Database) benchmark.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Scientific Principles & 2D Constraints](#2-scientific-principles--2d-constraints)
3. [Step-by-Step System Working & Pipeline Stages](#3-step-by-step-system-working--pipeline-stages)
4. [Environment Setup & Installation](#4-environment-setup--installation)
5. [How to Run Inference (Single Image & Batch Directory)](#5-how-to-run-inference)
6. [Training & Evaluation Workflows](#6-training--evaluation-workflows)
7. [Quantitative Benchmark Results](#7-quantitative-benchmark-results)
8. [Research Audit & Methodological Integrity](#8-research-audit--methodological-integrity)
9. [Repository Directory Layout](#9-repository-directory-layout)
10. [References & Acknowledgements](#10-references--acknowledgements)

---

## 1. Project Overview

Surface defect inspection is a vital quality assurance step in hot-rolled and cold-rolled steel manufacturing. Standard automated visual inspection systems typically stop at **binary classification** or **bounding box localization**. 

This project extends defect detection into **interpretable, quantitative defect severity estimation** by combining:
1. **Autonomous Object Detection**: Fast, high-accuracy localization across 6 industrial defect classes.
2. **Derived Weak Segmentation**: Bounding-box localized adaptive thresholding for contour extraction.
3. **Multi-Domain Feature Engineering**: 38 geometric, intensity, and GLCM/LBP texture descriptors.
4. **Continuous & Ordinal Severity Scoring**: Exploratory Geometric Severity Index (EGSI) and Random Forest/XGBoost models.
5. **Deep Representation Learning**: Pretrained EfficientNet-V2 dual-head neural network.
6. **Epistemic Uncertainty Quantification**: Monte Carlo Dropout ($N=25$) providing 95% Credible Intervals and automated quality control triage flags.
7. **Spatial Degradation Mapping**: Gradient-weighted Class Activation Mapping (Grad-CAM) thermal heatmaps.

---

## 2. Scientific Principles & 2D Constraints

To maintain metallurgical and scientific validity:

- **2D Grayscale Constraint**: NEU-DET images are $200 \times 200$ pixel 2D grayscale surface images. This pipeline **does not fabricate** 3D depth, crack depth, physical volume, or absolute material loss in physical units ($\text{mm}/\mu\text{m}$) without physical scale calibration. All measurements are reported in pixel space or normalized image coordinates.
- **Severity vs Confidence**: Defect severity is **never** equated to detector confidence. Severity is grounded in physical geometric characteristics (defect area ratio, elongation/aspect ratio, circularity, local contrast, GLCM/LBP texture).
- **Epistemic Uncertainty & Quality Triage**: Predictions near class boundaries are quantified using Monte Carlo Dropout standard deviation $\sigma$. High-uncertainty samples ($\sigma > \tau$) are automatically flagged for mandatory human metallurgist review.

---

## 3. Step-by-Step System Working & Pipeline Stages

```text
                               Raw Steel Surface Image
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
          [Stage 1: Detection]                       [Stage 1 (Alt): XML]
           YOLOv8n / YOLO11m                             Annotations
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                            [Stage 2: Derived Weak Mask]
                             Adaptive Otsu + Morphological
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
       [Stage 3A: Handcrafted]                       [Stage 3B: Deep Vision]
     38 Quantitative Descriptors                     EfficientNet-V2 Dual-Head
    (Geometric, Intensity, Texture)                             │
                   │                                           │
                   ▼                                           ▼
      [Exploratory Geometric Index]               [Epistemic Uncertainty & Grad-CAM]
      - Low / Moderate / High / Critical          - MC Dropout Mean (μ) ± 1.96*σ
      - RF & XGBoost Severity Models              - Thermal Degradation Heatmaps
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                        [Stage 4: Diagnostic Visualization]
                         4-Panel Multi-Modal Explanation
```

### Stage 1: Autonomous Object Detection
- The input steel image is passed through a trained YOLO detector (`models/detector_yolo.pt`).
- Predicts bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$, class identities, and confidence scores across 6 classes: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.

### Stage 2: Derived Weak Segmentation
- Crops the localized defect region of interest (ROI).
- Applies Gaussian smoothing, adaptive Otsu thresholding, and morphological opening/closing to isolate defect pixel masks.

### Stage 3A: Quantitative Handcrafted Feature Extraction
Extracts 38 mathematical descriptors per defect instance:
- **16 Geometric Metrics**: Defect area ($\text{px}^2$), area ratio ($\%$), perimeter, bounding box dimensions, aspect ratio, circularity, solidity, eccentricity, equivalent diameter, convexity, extent, principal moments.
- **8 Intensity Moments**: Mean, variance, standard deviation, skewness, kurtosis, local defect contrast, min/max intensity.
- **6 Texture Descriptors**: Gray-Level Co-occurrence Matrix (GLCM) contrast, dissimilarity, homogeneity, energy, correlation, and Local Binary Pattern (LBP) histogram entropy.

### Stage 3B: Deep Neural Severity & Uncertainty Estimation
- A cropped ROI tensor is passed to a dual-head **EfficientNet-V2** network.
- **Regression Head**: Predicts continuous severity score $\hat{S} \in [0, 100]$.
- **Classification Head**: Predicts 4 ordinal severity grades (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`).
- **Monte Carlo Dropout Engine**: Keeps dropout active during inference ($p=0.3$) and computes $N=25$ stochastic forward passes, generating predictive mean $\mu_S$, variance $\sigma_S^2$, and 95% Credible Interval $[\mu - 1.96\sigma, \mu + 1.96\sigma]$.

### Stage 4: Grad-CAM Thermal Severity Heatmaps
- Computes gradients of the predicted severity score w.r.t the final convolutional feature maps of EfficientNet-V2.
- Generates a 2D activation heatmap indicating exactly which pixel clusters drive the severity score.

### Stage 5: Multi-Modal Diagnostic Visualization & CSV Reporting
- Produces a **4-Panel Diagnostic Figure** for every detected defect:
  1. *Localization*: Original image with bounding box & confidence.
  2. *Contour*: Derived weak segmentation mask overlay.
  3. *Thermal Heatmap*: Grad-CAM spatial degradation map.
  4. *Quality Panel*: Quantitative metrics, EGSI score, neural score with $\pm \sigma$ uncertainty interval, and Human Triage Status badge.
- Appends all results to a structured `predictions_summary.csv` table.

---

## 4. Environment Setup & Installation

### Prerequisites
- Linux / macOS / Windows
- Python 3.10, 3.11, or 3.12
- GPU (NVIDIA CUDA) optional; CPU fully supported and vectorized.

### Step 1: Clone Repository
```bash
git clone https://github.com/Devil0510/Defect_Detector_AI.git
cd Defect_Detector_AI
```

### Step 2: Create Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate

# Install all required packages
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run System Check & Unit Tests
```bash
# Verify environment dependencies
python3 scripts/system_check.py

# Run unit tests
pytest tests/
```

---

## 5. How to Run Inference

### Scenario A: Process a Single Image
Run end-to-end multi-modal inference on any individual steel surface image:

```bash
python3 scripts/run_pipeline.py \
  --input_path data/raw/NEU-DET/IMAGES/patches_1.jpg \
  --detector_weights models/detector_yolo.pt \
  --neural_severity models/severity_efficientnet_v2.pt \
  --output_dir outputs/explanations
```

### Scenario B: Process an Entire Dataset / Directory of Images
When given a folder containing multiple unseen steel surface images:

```bash
python3 scripts/run_pipeline.py \
  --input_path /path/to/test_images_folder \
  --detector_weights models/detector_yolo.pt \
  --neural_severity models/severity_efficientnet_v2.pt \
  --output_dir outputs/evaluation_reports \
  --summary_csv outputs/predictions_summary.csv
```

#### Output Artifacts:
1. **Summary CSV (`outputs/predictions_summary.csv`)**:
   Contains columns: `image_name`, `defect_index`, `defect_class`, `detector_confidence`, `bbox_xmin`, `bbox_ymin`, `bbox_xmax`, `bbox_ymax`, `defect_area_px`, `area_ratio_pct`, `aspect_ratio`, `circularity`, `local_contrast`, `severity_score`, `severity_grade`, `neural_severity_mean`, `neural_uncertainty_std`, `requires_human_triage`, `diagnostic_report_path`.
2. **Visual Diagnostic Reports (`outputs/evaluation_reports/*_diagnostic.png`)**:
   High-resolution 4-panel visual reports for every defect instance.

---

## 6. Training & Evaluation Workflows

### 1. Dataset Inspection & Stratified Splitting
```bash
# Audit raw dataset and generate distribution plots
python3 scripts/inspect_dataset.py --data_dir data/raw/NEU-DET

# Create zero-leakage stratified splits (70% train / 15% val / 15% test)
python3 scripts/create_splits.py --data_dir data/raw/NEU-DET --seed 42

# Convert VOC XML annotations to YOLO format
python3 scripts/convert_annotations.py
```

### 2. YOLO Object Detector Training & Evaluation
```bash
# Train YOLO detector
python3 scripts/train_detector.py --model_variant yolov8n.pt --epochs 15 --batch_size 32

# Evaluate on held-out test split
python3 scripts/evaluate_detector.py --model_path models/detector_yolo.pt
```

### 3. Quantitative Feature Extraction & Severity Modeling
```bash
# Extract 38 geometric, intensity, and texture features across all defect instances
python3 scripts/extract_features.py

# Train Random Forest severity classifier on balanced quantile categories
python3 scripts/train_severity_model.py --model_type random_forest --output_model models/severity_rf.pkl

# Evaluate Random Forest severity model
python3 scripts/evaluate_severity.py --model_file models/severity_rf.pkl
```

### 4. EfficientNet-V2 Deep Severity & MC Dropout Evaluation
```bash
# Train EfficientNet-V2 neural severity model
python3 scripts/train_neural_severity.py --epochs 25 --batch_size 32

# Evaluate neural severity with Monte Carlo Dropout uncertainty estimation
python3 scripts/evaluate_neural_severity.py --model_path models/severity_efficientnet_v2.pt
```

### 5. Feature Group Ablation Study
```bash
python3 scripts/run_ablation.py
```

---

## 7. Quantitative Benchmark Results

### A. Object Detection Performance (Held-Out Test Set: 269 images, 643 defects)

| Detection Metric | Result |
| :--- | :---: |
| **mAP@0.5** | **0.6898 (69.0%)** |
| **mAP@0.5:0.95** | **0.3823 (38.2%)** |
| **Mean Precision** | **0.8002 (80.0%)** |
| **Mean Recall** | **0.5929 (59.3%)** |
| **Inference Latency** | **5.65 ms / image** (>175 FPS) |

#### Per-Class AP@0.5:
- **Patches (`patches`)**: **0.884 (88.4%)**
- **Scratches (`scratches`)**: **0.858 (85.8%)**
- **Pitted Surface (`pitted_surface`)**: **0.833 (83.3%)**
- **Inclusion (`inclusion`)**: **0.786 (78.6%)**
- **Rolled-in Scale (`rolled-in_scale`)**: **0.516 (51.6%)**
- **Crazing (`crazing`)**: **0.261 (26.1%)**

---

### B. Severity Estimation & Uncertainty Metrics (Held-Out Test Set: 644 defects)

| Model / Methodology | Metric | Test Set Result |
| :--- | :--- | :---: |
| **Random Forest (Handcrafted 38 Descriptors)** | Accuracy / Macro F1 | **95.34% / 0.9510** |
| | Quadratic Weighted Kappa | **0.9821** |
| | Critical Defect Recall | **99.0% (140 / 142)** |
| **EfficientNet-V2 Neural Model (Deep Vision)** | Continuous Score MAE (out of 100) | **3.93 points** |
| | Pearson Correlation ($r$) | **0.9649** |
| | Spearman Rank Correlation ($\rho$) | **0.9635** |
| | Mean Epistemic Uncertainty ($\sigma$) | **±2.71 points** |
| | Human Quality Triage Rate | **0.9% of samples flagged** |

---

## 8. Research Audit & Methodological Integrity

A comprehensive 9-report scientific audit is documented in [`reports/research_audit/`](reports/research_audit/README.md):
- **Surrogate vs Ground Truth**: Handcrafted ML classifiers learn surrogate emulation of the continuous geometric severity index formula rather than independent physical truth.
- **Quantile Recalibration**: Severity category thresholds use empirical quantile boundaries ($S < 25$ Low, $25 \le S < 45$ Moderate, $45 \le S < 62$ High, $S \ge 62$ Critical), resolving class imbalance and boosting Critical recall from 20% to 99%.
- **2D Geometric Proxy**: The severity index is an exploratory geometric formula computed from 2D pixel measurements. It is not derived from industrial metallographic standards that require 3D depth gauges or physical instruments. All scores are documented as research proxies, not ground-truth physical severity.

---

## 9. Repository Directory Layout

```
Defect_Detector_AI/
├── README.md                    # Master documentation & guide
├── requirements.txt             # Python package dependencies
├── pyproject.toml               # Project metadata & pytest configuration
├── .gitignore                   # Git exclusion rules
├── configs/                     # YAML configuration files
│   ├── dataset.yaml
│   ├── detection.yaml
│   ├── segmentation.yaml
│   ├── severity.yaml
│   └── yolo_dataset.yaml
├── data/
│   ├── raw/NEU-DET/             # Original 1,770 images & XML annotations
│   ├── interim/labels/          # Converted YOLO text annotations
│   ├── interim/yolo/            # Standardized YOLO train/val/test splits
│   ├── processed/               # defect_features.csv (38 extracted descriptors)
│   └── splits/                  # Zero-leakage train/val/test split files
├── src/
│   ├── data/                    # XML parser, validator, stratified splitter
│   ├── detection/               # YOLO training wrappers and inference adapter
│   ├── segmentation/           # Weak ROI adaptive Otsu segmenter
│   ├── features/                # 38 geometric, intensity, and texture extractors
│   ├── severity/                # Severity index, ML models, EfficientNet-V2, MC Dropout
│   ├── evaluation/              # Ordinal classification and regression metrics
│   └── visualization/          # Multi-panel explainer and Grad-CAM heatmap engine
├── scripts/
│   ├── system_check.py          # Environment diagnostics
│   ├── inspect_dataset.py       # Dataset audit and distribution plots
│   ├── create_splits.py         # Deterministic split generator
│   ├── convert_annotations.py   # XML to YOLO converter
│   ├── train_detector.py        # YOLO object detector trainer
│   ├── evaluate_detector.py     # YOLO test split evaluator
│   ├── extract_features.py      # Quantitative feature extraction pipeline
│   ├── train_severity_model.py  # Random Forest / XGBoost trainer
│   ├── evaluate_severity.py     # Severity test evaluator
│   ├── train_neural_severity.py # EfficientNet-V2 deep severity trainer
│   ├── evaluate_neural_severity.py # Neural severity & uncertainty evaluator
│   ├── run_ablation.py          # Feature group ablation study
│   └── run_pipeline.py          # Unified autonomous end-to-end inference CLI
├── models/                      # Trained model checkpoints (.pt, .pkl)
├── reports/                     # Quantitative JSON benchmark reports & figures
│   └── research_audit/          # 9-report scientific methodological audit
├── outputs/                     # Visual explanation reports & predictions summary CSV
└── tests/                       # Pytest unit test suite
```

---

## 10. References & Acknowledgements

1. **NEU-DET Dataset**: K. Song and Y. Yan, *"A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defect detection,"* Applied Surface Science, vol. 285, pp. 858-864, 2013.
2. **YOLO Architecture**: J. Redmon, A. Farhadi, et al., Ultralytics YOLOv8/YOLO11 framework.
3. **EfficientNet-V2**: M. Tan and Q. V. Le, *"EfficientNetV2: Smaller Models and Faster Training,"* ICML 2021.
4. **Monte Carlo Dropout**: Y. Gal and Z. Ghahramani, *"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,"* ICML 2016.
5. **Grad-CAM**: R. R. Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,"* ICCV 2017.
