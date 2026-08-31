# Object Detection Stage Audit

## 1. Detection Model Implementation Status

```text
Detection stage incomplete.
```

### Current Status Breakdown:
- **Annotation Conversion**: Completed. Pascal VOC XML annotations were converted to normalized YOLO format (`<class_id> <x_center> <y_center> <w> <h>`) in `data/interim/labels/` (1,770 files, 4,126 boxes).
- **Detector Training (`scripts/train_detector.py`)**: **Not yet executed.** No YOLO weights (e.g. `best.pt`, `last.pt`) have been generated or saved in `models/`.
- **Detector Test Evaluation (`scripts/evaluate_detector.py`)**: **Not yet evaluated.** Detection metrics (mAP@0.5, mAP@0.5:0.95, per-class AP, precision-recall curves, inference latency) have not been produced from a trained neural detector.
- **Current Severity Pipeline Localization Source**: The feature extraction and pipeline scripts currently utilize the **ground truth XML annotations** (Pascal VOC bounding boxes) as the defect localization input rather than predictions from an autonomous deep learning detector.

---

## 2. Detection Configuration Requirements (For Phase 2 Execution)

To complete the detection stage with scientific rigor, the following parameters must be formally trained and logged:

| Parameter | Planned Specification | Notes |
| :--- | :--- | :--- |
| **Model Architecture** | YOLOv8n / YOLOv11n (or YOLOv8s) | Scaled for NVIDIA T600 (4GB VRAM) constraints. |
| **Input Image Size (`imgsz`)** | $200 \times 200$ (native) vs $640 \times 640$ (upscaled) | Requires controlled comparison on 200x200 NEU-DET. |
| **Training Split** | `data/splits/train.txt` (1,237 images) | 70% stratified training split. |
| **Validation Split** | `data/splits/val.txt` (264 images) | 15% validation split for early stopping. |
| **Held-Out Test Split** | `data/splits/test.txt` (269 images) | 15% held-out test split for final evaluation only. |
| **Random Seed** | `42` | Deterministic initialization. |
| **Confidence Threshold** | Configurable (Default $0.25$) | Tested across $[0.10, 0.70]$. |
| **NMS IoU Threshold** | Configurable (Default $0.45$) | Evaluated for overlapping defect boxes. |

---

## 3. Metrics Required to Validate Detection
Before claiming a complete end-to-end vision system:
1. **Precision, Recall, F1** at optimal confidence threshold.
2. **mAP@0.5** (PASCAL VOC metric).
3. **mAP@0.5:0.95** (COCO metric).
4. **Per-Class AP** across all 6 classes (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`).
5. **Detector Inference Latency** (ms per image on CPU vs GPU).
6. **Detector-Feature Error Propagation**: Analysis of how detector localization errors (imperfect bounding box IoU) impact downstream feature extraction and severity scoring.
