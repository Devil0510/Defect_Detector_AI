# Client User Guide: AI-Based Steel Surface Defect Severity Estimation

Welcome to the **AI-Based Steel Surface Defect Severity Estimation System**. This enterprise-ready system is designed for quality control (QC) engineers, metallurgists, plant managers, and automated inspection teams in steel manufacturing.

This guide explains **how to use the system**, **which AI models and algorithms are used and why**, **what parameters are available for configuration**, and **how to interpret the diagnostic output reports**.

---

## Table of Contents

1. [Executive Summary & Business Value](#1-executive-summary--business-value)
2. [Architectural Overview: Models Used & Why](#2-architectural-overview-models-used--why)
3. [System Parameters Reference Guide](#3-system-parameters-reference-guide)
4. [Step-by-Step Usage Guide](#4-step-by-step-usage-guide)
   - [A. Environment Setup (One-Time)](#a-environment-setup-one-time)
   - [B. Inspecting a Single Image](#b-inspecting-a-single-image)
   - [C. Batch Processing an Entire Folder / Shift Batch](#c-batch-processing-an-entire-folder--shift-batch)
   - [D. Interactive Mode (No Command Line Experience Required)](#d-interactive-mode-no-command-line-experience-required)
5. [Understanding the Output Reports](#5-understanding-the-output-reports)
   - [A. The 4-Panel Visual Diagnostic Report](#a-the-4-panel-visual-diagnostic-report)
   - [B. Summary CSV Metrics & Column Definitions](#b-summary-csv-metrics--column-definitions)
   - [C. Quality Control (QC) Triage Actions](#c-quality-control-qc-triage-actions)
6. [Troubleshooting & Frequently Asked Questions](#6-troubleshooting--frequently-asked-questions)

---

## 1. Executive Summary & Business Value

Traditional Automated Optical Inspection (AOI) systems typically provide only **binary detection** ("Defect Found" vs. "Clean") or basic bounding boxes. They do not tell you **how severe** the damage is, **how confident** the AI is, or **which physical attributes** make it dangerous.

This system bridges that gap by providing:
- **Millisecond-Grade Defect Localization**: Pinpoints up to 6 distinct defect classes on hot-rolled and cold-rolled steel strips.
- **Physical & Neural Severity Scoring**: Scores every defect on a continuous **0–100 Severity Index** and categorizes it into 4 actionable grades: `LOW`, `MODERATE`, `HIGH`, or `CRITICAL`.
- **Epistemic Uncertainty & Conformal Guarantees**: Mathematically quantifies how sure the model is. If an ambiguous defect is near a grading boundary, the system automatically flags it for human review.
- **Visual Explainability (Grad-CAM)**: Thermal gradient heatmaps highlight exactly which pixel clusters triggered the severity score.

---

## 2. Architectural Overview: Models Used & Why

The pipeline combines complementary computer vision and machine learning technologies:

```text
  Raw Steel Image
        │
        ▼
[ 1. YOLO Detection Engine ] ──► Fast Bounding Box Localization (6 classes)
        │
        ▼
[ 2. Weak ROI Segmentation ] ──► Isolates Defect Contours (Adaptive Otsu)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
[ 3. Quantitative Geometry ]      [ 4. EfficientNet-V2 Dual-Head ]
  - 38 Physical Descriptors         - Continuous Severity (0-100)
  - Metallurgical Weighting          - Ordinal Class (Low/Mod/High/Crit)
        │                                  │
        ▼                                  ▼
[ 5. Decision Tree Baseline ]     [ 5. Uncertainty & Explainability ]
  - Random Forest / XGBoost         - Monte Carlo Dropout (N=25 passes)
                                    - Conformal 95% Coverage Intervals
                                    - Grad-CAM Thermal Activation Maps
                                           │
                                           ▼
                                 [ 6. QC Triage Decision ]
                                  ✅ High Confidence: Auto-Pass/Reject
                                  ⚠️ Low Confidence: Route to Metallurgist
```

### Why These Specific Models Were Chosen:

| Model / Algorithm | Role in Pipeline | Why It Was Selected |
| :--- | :--- | :--- |
| **YOLOv8n / YOLO11m** *(Ultralytics)* | Autonomous Object Detection | **Extreme Speed (<6 ms per image, >175 FPS)**. Handles multi-scale defects ranging from hairline scratches to wide oxidation patches with high mean precision (80.0%) and zero reliance on manual cropping. |
| **Weak ROI Segmenter** *(Adaptive Otsu + Morphology)* | Defect Boundary Isolation | Extracts precise pixel masks from bounding boxes without requiring expensive manual polygon annotations, enabling accurate 2D area and perimeter measurements. |
| **Defect-Specific Geometric Weighting** | Physical Damage Index (EGSI) | Bridges pure deep learning with **metallurgical domain knowledge**. For instance, linear cracks (*scratches*) prioritize length and aspect ratio, while *pitted surfaces* prioritize texture and roughness. |
| **EfficientNet-V2-S Dual-Head** *(Deep CNN)* | Neural Representation & Severity Scoring | Highly parameter-efficient deep neural network with fused-MBConv layers. The **dual-head** architecture simultaneously predicts a continuous severity score $[0, 100]$ (MAE = 3.93 pts) and discrete sorting tiers (`Low`, `Moderate`, `High`, `Critical`). |
| **Monte Carlo Dropout ($N=25$)** | Epistemic Uncertainty Estimation | Standard deep networks are notoriously overconfident on borderline cases. Keeping dropout active at inference computes predictive variance ($\sigma$), exposing whether the model is guessing or certain. |
| **Split Conformal Prediction** | Statistically Sound Prediction Intervals | Unlike heuristic error bars, conformal prediction provides a **mathematical guarantee of 95% coverage** on future unseen production batches. |
| **Temperature Scaling** | Probability Calibration | Calibrates softmax confidence scores so that a predicted 90% probability corresponds to a true 90% historical accuracy rate. |
| **Grad-CAM Heatmap Engine** | Visual Explainability | Creates thermal heatmaps showing the gradient flow of severity. Essential for quality audits, regulatory compliance, and operator trust. |
| **Random Forest Classifier** | Feature-Based Surrogate Model | Acts as a fast, interpretable rule-verification check achieving **99.0% Critical Defect Recall** based on 38 extracted physical descriptors. |

---

## 3. System Parameters Reference Guide

All parameters have sensible industrial defaults, but can be tailored to match your plant's specific inspection tolerance:

### Command-Line Parameters

| Parameter | Flag | Default | Description & Industrial Guidance |
| :--- | :--- | :---: | :--- |
| **Input Path** | `--input_path` / `-i` | *Required* | Path to a single image file (`.jpg`, `.png`, `.bmp`, `.tif`) **OR** a folder containing multiple batch images. |
| **Output Directory** | `--output_dir` / `-o` | `outputs/` | Target folder where 4-panel diagnostic reports and summary CSVs will be saved. |
| **Summary CSV** | `--summary_csv` | `outputs/summary.csv` | Path to save the quantitative defect metrics table. |
| **YOLO Weights** | `--detector_weights` | `models/detector_yolo11m.pt` | Path to detection weights (`detector_yolo.pt` for nano, `detector_yolo11m.pt` for medium). |
| **Neural Severity Weights** | `--neural_severity` | `models/severity_efficientnet_v2.pt` | Path to the trained EfficientNet-V2 dual-head checkpoint. |
| **Conformal Calibrator** | `--conformal_calibrator` | `models/conformal_calibrator.json` | Calibration coefficients providing 95% coverage guarantees. |
| **Temperature Scaler** | `--temperature_scaler` | `models/temperature_scaler.json` | Probability calibration model for ordinal classification logits. |
| **Detection Confidence** | `--conf_threshold` | `0.20` | Minimum detector confidence to register a defect. **Lower (e.g. 0.15)** to catch extremely faint defects; **raise (e.g. 0.40)** to minimize false alarms on noisy background steel. |
| **CLAHE Enhancement** | `--use_clahe` | `False` | Apply Contrast Limited Adaptive Histogram Equalization. **Recommended for low-contrast or poorly lit coils** to highlight micro-fissures (*crazing*). |
| **Recursive Scan** | `--recursive` / `-r` | `False` | Recursively scan nested subfolders in batch analysis. |
| **Hardware Device** | `--device` | Auto (`cuda` if available, else `cpu`) | Execution target. Enter `cpu` or `cuda`. Automatic fallback is included if GPU driver versions mismatch. |

---

### Internal Quality Triage Thresholds

These thresholds govern whether a defect passes automatically or routes to human inspectors:

| Metric | Variable | Threshold | Action Triggered |
| :--- | :--- | :---: | :--- |
| **Detector Confidence** | `min_detector_confidence` | `< 50.0%` | Flagged: Model is uncertain if a defect genuinely exists. |
| **Epistemic Uncertainty** | `max_mc_std` ($\sigma$) | `> 8.0 points` | Flagged: Model exhibits high disagreement across Monte Carlo forward passes. |
| **Conformal Interval Width** | `max_conformal_width` ($\Delta$) | `> 30.0 points` | Flagged: 95% confidence interval is too wide to guarantee defect grade. |
| **Predictive Entropy** | `entropy` | `> 1.1 nats` | Flagged: Probability is dispersed across multiple severity tiers. |

---

## 4. Step-by-Step Usage Guide

### A. Environment Setup (One-Time)

Open your terminal in the project directory:

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Verify all models and dependencies are working properly
pytest tests/
```
*Expected output: `12 passed in ~3.0s`.*

---

### B. Inspecting a Single Image

To run full multi-modal inspection on an individual steel surface image:

```bash
python3 scripts/run_pipeline.py \
  --input_path data/raw/NEU-DET/IMAGES/patches_1.jpg \
  --detector_weights models/detector_yolo.pt \
  --neural_severity models/severity_efficientnet_v2.pt \
  --output_dir outputs/single_inspection
```

**Console Output:**
```text
======================================================================
DEFECT #1 DIAGNOSTIC INFERENCE — patches_1.jpg
======================================================================
Detected Class      : patches (Confidence: 89.4%)
Bounding Box        : [14.0, 112.0, 185.0, 195.0]
Geometric Severity  : 68.4 / 100 (CRITICAL)
Neural Severity (μ) : 66.8 ± 2.31 points
95% Confidence Int. : [61.2, 72.4] (Conformal Calibrated 95%) (Width: 11.2)
QC / Triage Status  : ✅ HIGH_CONFIDENCE
Diagnostic Report   : outputs/single_inspection/patches_1_defect_001_diagnostic.png
```

---

### C. Batch Processing an Entire Folder / Shift Batch

When analyzing an entire directory of incoming inspection images from a production shift:

```bash
python3 scripts/batch_analysis.py \
  --input_dir /path/to/incoming_shift_images \
  --output_dir outputs/shift_batch_results \
  --use_clahe \
  --yes
```

**What the batch runner produces:**
1. **Per-Specimen Folders**: A dedicated folder for each inspected image containing:
   - `specimen_summary.json` (Machine-readable metadata for MES/SCADA integration).
   - `specimen_summary.csv` (Defect table for this specimen).
   - `diagnostics/*_diagnostic.png` (4-panel visual reports for every defect found).
2. **Master Summary Table (`MASTER_BATCH_SUMMARY.csv`)**: Consolidates every single defect across all specimens in the batch.
3. **Specimen Level Rollup (`SPECIMEN_SUMMARY.csv`)**: One row per image showing total defects, dominant defect type, peak severity, and overall pass/triage status.

---

### D. Interactive Mode (No Command Line Experience Required)

Operators can run the batch tool without remembering any CLI flags:

```bash
python3 scripts/batch_analysis.py
```

The system will interactively prompt:
```text
============================================================
AI SURFACE DEFECT SEVERITY ESTIMATION SYSTEM
BATCH ANALYSIS MODE
============================================================

Enter path to input image folder:
> /home/user/incoming_steel_plates

Enter path where results should be saved:
> /home/user/inspection_reports

Discovered 150 supported image(s).
Proceed with batch analysis? [Y/n]: y
```

---

## 5. Understanding the Output Reports

### A. The 4-Panel Visual Diagnostic Report

Every detected defect generates a high-resolution 4-panel image:

```text
┌─────────────────┬─────────────────┬─────────────────┬─────────────────────────┐
│  PANEL 1        │  PANEL 2        │  PANEL 3        │  PANEL 4                │
│  Localization   │  Weak Mask      │  Grad-CAM       │  Quantitative Metrics   │
│                 │                 │                 │                         │
│  [Original +    │  [Green Contour │  [Thermal Heat- │  - Defect Area & Ratio  │
│   Blue BBox +   │   Extracted via │   map Highligh- │  - Aspect Ratio         │
│   Confidence]   │   Adaptive      │   ting Severity │  - Local & GLCM Contrast│
│                 │   Otsu]         │   Clusters]     │  - Severity: 68.4 / 100 │
│                 │                 │                 │  - Grade: CRITICAL      │
│                 │                 │                 │  - 95% Conf. Int.       │
│                 │                 │                 │  - [QC STATUS: PASS]    │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
```

1. **Panel 1 (Localization)**: Shows the raw steel surface with the bounding box and classification confidence.
2. **Panel 2 (Derived Weak Mask)**: Highlights the exact morphological defect contours in green to inspect surface penetration.
3. **Panel 3 (Grad-CAM Severity Heatmap)**: Uses a thermal color spectrum (Blue = cold/normal, Red = hot/destructive) indicating which internal regions triggered high severity.
4. **Panel 4 (QC Metrics Badge)**: Displays mathematical geometric descriptors, the physical severity score, the deep neural predictive score ($\mu \pm \sigma$), and the color-coded Quality Control triage badge.

---

### B. Summary CSV Metrics & Column Definitions

The output CSV files contain structured engineering data for database logging or statistical process control (SPC):

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `image_name` | String | Filename of the steel image. |
| `defect_index` | Integer | Defect occurrence index on this image (1, 2, ...). |
| `defect_class` | String | Classified category (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`). |
| `detector_confidence` | Float (0–1) | YOLO model confidence in the localization. |
| `bbox_xmin`, `ymin`, `xmax`, `ymax` | Float | Spatial pixel coordinates of the bounding box. |
| `defect_area_px` | Float | Measured surface damage area in square pixels ($\text{px}^2$). |
| `area_ratio_pct` | Float (%) | Percentage of total steel sheet surface occupied by this defect. |
| `aspect_ratio` | Float | Ratio of defect length to width (high values indicate elongated crack risks). |
| `circularity` | Float (0–1) | Degree of circular compactness ($1.0 = \text{perfect circle}$, $<0.2 = \text{fissure}$). |
| `local_contrast` | Float (0–1) | Relative brightness difference between defect and surrounding steel. |
| `severity_score` | Float (0–100)| Continuous rule-based geometric severity rating. |
| `severity_grade` | String | Discrete sorting grade (`Low`, `Moderate`, `High`, `Critical`). |
| `neural_severity_mean` | Float (0–100)| Dual-head EfficientNet-V2 predictive mean ($\mu$). |
| `neural_uncertainty_std` | Float ($\sigma$) | Monte Carlo Dropout standard deviation (predictive uncertainty). |
| `ci_95_lower`, `ci_95_upper` | Float | Lower and upper bounds of the 95% Conformal Prediction Interval. |
| `conformal_interval_width` | Float | Total width of the 95% interval ($\Delta = \text{Upper} - \text{Lower}$). |
| `qc_status` | String | `HIGH_CONFIDENCE`, `MODERATE_CONFIDENCE`, or `HUMAN_REVIEW_REQUIRED`. |
| `requires_human_triage` | Boolean | `True` if manual metallurgist inspection is mandatory. |
| `diagnostic_report_path` | String | Absolute filepath to the visual 4-panel report PNG. |

---

### C. Quality Control (QC) Triage Actions

The system automatically assigns one of three statuses to every inspected defect:

- 🟢 **`HIGH_CONFIDENCE`**:
  - **Meaning**: The model is confident in both detection and severity classification.
  - **Action**: Automated line action can proceed (e.g. coil auto-passed for prime sale or auto-routed for trimming).
- 🔵 **`MODERATE_CONFIDENCE`**:
  - **Meaning**: The defect was detected reliably, but parameters are near a boundary tier.
  - **Action**: System logs the event; can be sampled in periodic quality audits.
- 🔴 **`HUMAN_REVIEW_REQUIRED`**:
  - **Meaning**: Uncertainty standard deviation is elevated ($\sigma > 8.0$), confidence is low ($< 50\%$), or the 95% interval spans across multiple severity tiers.
  - **Action**: **Mandatory Human Metallurgist Review**. The inspector should open the corresponding 4-panel diagnostic PNG, inspect the thermal heatmap, and confirm the grade.

---

## 6. Troubleshooting & Frequently Asked Questions

### Q: Why does the system report "CUDA initialization failed, falling back to CPU"?
**A**: This occurs if your NVIDIA driver version does not match the PyTorch CUDA build. The pipeline includes an automated fallback that safely switches execution to the CPU. All models (YOLO, EfficientNet-V2, and Random Forest) run reliably on multi-core CPUs.

### Q: Why are defect dimensions reported in pixels rather than millimeters ($\text{mm}$)?
**A**: To maintain scientific and metallurgical integrity, physical units ($\text{mm}$ or $\mu\text{m}$) cannot be computed without knowing the camera lens focal length and physical distance to the steel strip (pixel-to-millimeter ratio). Once your camera system is calibrated, pixel values can be multiplied by your plant's calibration factor.

### Q: What should I do if faint surface cracks (*crazing*) are being missed?
**A**: Enable the `--use_clahe` flag and lower the detection threshold:
```bash
python3 scripts/batch_analysis.py -i /path/to/images -o outputs/test --use_clahe --conf_threshold 0.15
```

### Q: How do I integrate the output with our Plant MES / SCADA Database?
**A**: Read the generated `MASTER_BATCH_SUMMARY.csv` or poll the per-specimen `specimen_summary.json` files using your plant's data ingestion pipeline (e.g., Python, C#, or SQL loader).

---
*For further technical inquiries, model retraining instructions, or custom camera calibrations, refer to the master technical documentation in [`README.md`](README.md) and research audits in [`reports/research_audit/`](reports/research_audit/).*
