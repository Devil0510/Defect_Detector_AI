# AI-Based Steel Surface Defect Severity Estimation

A research-grade, reproducible, modular computer-vision and machine-learning pipeline for steel surface defect detection, quantitative feature extraction, deep neural severity estimation, Monte Carlo Dropout epistemic uncertainty quantification, and spatial Grad-CAM severity mapping using the NEU-DET (Northeastern University Surface Defect Database) benchmark.

---

## 1. Scientific Objectives & Rigor

- **Objective**: Autonomously detect steel surface defects, identify defect classes, extract quantitative 2D geometric and texture descriptors, calculate continuous and ordinal severity scores, quantify predictive uncertainty, and generate spatial heatmaps for industrial quality control.
- **Scientific Limitation (2D Surface Imagery)**: NEU-DET images are $200 \times 200$ pixel 2D grayscale surface images. This pipeline **does not** fabricate 3D depth, crack depth, physical volume, or absolute material loss in physical units (mm/$\mu\text{m}$) without physical scale calibration. All measurements are reported in pixel space or normalized image coordinates.
- **Severity vs Confidence**: Defect severity is **never** equated to detector confidence. Severity is grounded in physical and geometric characteristics (area, aspect ratio, circularity, local contrast, GLCM/LBP texture).
- **Epistemic Uncertainty & Quality Triage**: Employs Monte Carlo Dropout ($N=25$) to provide predictive standard deviation $\sigma$ and 95% Credible Intervals. Boundary cases with $\sigma > \tau$ are automatically flagged for human metallurgist review.

---

## 2. System Architecture

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

---

## 3. Quantitative Benchmark Results

### A. Autonomous Object Detection (Held-Out Test Set: 269 images, 643 defects)

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n (Default)** | **0.6898 (69.0%)** | **0.3823 (38.2%)** | **0.8002 (80.0%)** | **0.5929 (59.3%)** | **5.65 ms / image** |

#### Per-Class Detection AP@0.5:
- **Patches (`patches`)**: **0.884 (88.4%)**
- **Scratches (`scratches`)**: **0.858 (85.8%)**
- **Pitted Surface (`pitted_surface`)**: **0.833 (83.3%)**
- **Inclusion (`inclusion`)**: **0.786 (78.6%)**
- **Rolled-in Scale (`rolled-in_scale`)**: **0.516 (51.6%)**
- **Crazing (`crazing`)**: **0.261 (26.1%)**

---

### B. Severity Estimation & Uncertainty Benchmarks

| Methodology | Metric | Test Set Result |
| :--- | :--- | :---: |
| **Random Forest (Handcrafted 38 Features)** | Accuracy / Macro F1 | **95.34% / 0.9510** |
| | Quadratic Weighted Kappa | **0.9821** |
| | Critical Defect Recall | **99.0% (140 / 142)** |
| **EfficientNet-V2 Neural Severity Model** | Score MAE (out of 100) | **4.12 points** |
| | Spearman Rank Correlation ($\rho$) | **0.9240** |
| | 95% Credible Interval Coverage | **94.8%** |
| | Mean Epistemic Uncertainty ($\sigma$) | **±3.41 points** |

---

## 4. Repository Structure

```
Defect_Detector_AI/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── dataset.yaml
│   ├── detection.yaml
│   ├── segmentation.yaml
│   ├── severity.yaml
│   └── yolo_dataset.yaml
├── data/
│   ├── raw/NEU-DET/             # Original 1,770 images & XML annotations
│   ├── interim/yolo/            # Standardized YOLO train/val/test splits
│   ├── processed/               # defect_features.csv (38 descriptors)
│   └── splits/                  # Stratified zero-leakage split files
├── src/
│   ├── data/                    # XML parser, dataset validator, split generator
│   ├── detection/               # YOLO training wrappers and inference adapter
│   ├── segmentation/           # Weak ROI adaptive threshold segmenter
│   ├── features/                # 38 geometric, intensity, and GLCM/LBP texture extractors
│   ├── severity/                # Severity index, ML models, EfficientNet-V2, MC Dropout
│   └── visualization/          # Explainer, Grad-CAM heatmaps, distribution plots
├── scripts/
│   ├── system_check.py          # System environment diagnostics
│   ├── inspect_dataset.py       # Dataset audit & distribution plotting
│   ├── create_splits.py         # Deterministic stratified split generator
│   ├── convert_annotations.py   # VOC XML to YOLO converter
│   ├── train_detector.py        # YOLO object detector trainer
│   ├── evaluate_detector.py     # YOLO test split evaluator
│   ├── extract_features.py      # Quantitative feature extraction pipeline
│   ├── train_severity_model.py  # Random Forest / XGBoost severity trainer
│   ├── evaluate_severity.py     # Severity test evaluator
│   ├── train_neural_severity.py # EfficientNet-V2 deep severity trainer
│   ├── evaluate_neural_severity.py # Neural severity & uncertainty evaluator
│   ├── run_ablation.py          # Feature group ablation study
│   └── run_pipeline.py          # Unified autonomous end-to-end inference CLI
├── models/                      # Trained weights (detector_yolo.pt, severity_rf.pkl, etc.)
├── reports/                     # Benchmark JSON reports and figures
│   └── research_audit/          # 9-report rigorous methodological audit
└── tests/                       # Pytest unit test suite
```

---

## 5. Quick Start & Execution

### 1. Installation

```bash
git clone https://github.com/Devil0510/Defect_Detector_AI.git
cd Defect_Detector_AI

# Install dependencies
python3 -m pip install --user --break-system-packages -r requirements.txt
```

### 2. Run Test Suite

```bash
pytest tests/
```

### 3. Run Autonomous End-to-End Inference

Execute the complete multi-modal pipeline (Detection $\to$ Derived Segmentation $\to$ Quantitative Metrics $\to$ Severity $\to$ Uncertainty $\to$ Grad-CAM Heatmap):

```bash
python3 scripts/run_pipeline.py \
  --image_path data/raw/NEU-DET/IMAGES/patches_1.jpg \
  --detector_weights models/detector_yolo.pt \
  --neural_severity models/severity_efficientnet_v2.pt
```

### 4. Re-train Detector or Severity Models

```bash
# Train YOLO detector
python3 scripts/train_detector.py --model_variant yolov8n.pt --epochs 15 --batch_size 32

# Train EfficientNet-V2 Neural Severity Model
python3 scripts/train_neural_severity.py --epochs 25 --batch_size 32

# Run Feature Ablation Study
python3 scripts/run_ablation.py
```

---

## 6. Citation & Dataset Acknowledgement

- **Dataset**: Northeastern University (NEU) Surface Defect Database (K. Song & Y. Yan, *Sensors*, 2013).
- **License**: MIT License.
