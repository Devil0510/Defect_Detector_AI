# Claims & Empirical Evidence Audit

## 1. Feature Group Ablation Reality Check

The feature ablation study conducted in `scripts/run_ablation.py` demonstrated the following progression:
- **Exp 1 (BBox Dimensions Only)**: 86.65% Accuracy, 0.1335 MAE
- **Exp 2 (Geometry Features)**: 95.65% Accuracy, 0.0435 MAE (+9.0% accuracy gain)
- **Exp 3 (Geometry + Intensity)**: 95.81% Accuracy, 0.0419 MAE
- **Exp 4 (Full Features)**: 96.12% Accuracy, 0.0388 MAE
- **Exp 5 (XGBoost Architecture)**: 97.98% Accuracy, 0.0202 MAE

### Mandatory Scientific Distinction:
> **The ablation demonstrates feature importance for reproducing the constructed severity index, not necessarily for predicting independently validated physical severity.**

Because the ground truth labels ($y$) are derived from the feature calculation $S(A, L, C, T)$, adding those exact feature dimensions (geometric area, length, contrast, texture) directly provides the model with the arithmetic terms of the target equation.

---

## 2. Severity Class Distribution & Class Imbalance Audit

### Empirical Class Breakdown:

| Partition | Total Samples | Low ($S < 25$) | Moderate ($25 \le S < 50$) | High ($50 \le S < 75$) | Critical ($S \ge 75$) | Imbalance Ratio (Max : Min) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Set** | 2,888 | 748 (25.90%) | 880 (30.47%) | 1,248 (43.21%) | **12 (0.42%)** | **104 : 1** |
| **Val Set** | 594 | 126 (21.21%) | 211 (35.52%) | 255 (42.93%) | **2 (0.34%)** | **127 : 1** |
| **Test Set** | 644 | 203 (31.52%) | 196 (30.43%) | 240 (37.27%) | **5 (0.78%)** | **48 : 1** |
| **Complete Dataset**| 4,126 | 1,077 (26.10%) | 1,287 (31.19%) | 1,743 (42.24%) | **19 (0.46%)** | **92 : 1** |

### Detailed Test Confusion Matrix & Minority Class Failure:

#### XGBoost Model Breakdown on Held-Out Test Set (644 Defect Instances):
```text
               precision    recall  f1-score   support
         Low        1.00      0.99      0.99       203
    Moderate        0.98      0.97      0.98       196
        High        0.97      0.99      0.98       240
    Critical        1.00      0.20      0.33         5

    accuracy                            0.98       644
   macro avg        0.99      0.79      0.82       644
weighted avg        0.98      0.98      0.98       644
```

#### Test Confusion Matrix:
$$\begin{pmatrix} 201 & 2 & 0 & 0 \\ 1 & 191 & 4 & 0 \\ 0 & 2 & 238 & 0 \\ 0 & 0 & 4 & 1 \end{pmatrix}$$

### Critical Diagnostic Finding:
- **97.98% overall accuracy is misleading**: It is dominated by Low, Moderate, and High classes.
- For the **Critical class**, the model achieved only **20.0% recall (1 out of 5 detected)**, with 4 critical defects misclassified as High.

---

## 3. Feature-Target Correlation Analysis

| Feature Descriptor | Feature Modality | Pearson Correlation ($r$) | Spearman Rank Correlation ($\rho$) | Relationship to Target Formulation |
| :--- | :--- | :---: | :---: | :--- |
| `defect_area_px` | Derived Geometry | **+0.7061** | **+0.9490** | **Direct input component** ($w_A = 0.40$). Strongest driver. |
| `bbox_area_px` | Bounding Box | +0.7060 | +0.9410 | Highly collinear with `defect_area_px` ($r > 0.95$). |
| `length_px` | Bounding Box / Extent | **+0.7961** | **+0.8184** | **Direct input component** ($w_L = 0.25$). |
| `glcm_contrast` | GLCM Texture | **+0.3369** | **+0.3592** | **Direct input component** ($w_T = 0.15$). |
| `local_contrast` | Intensity Moment | **+0.1254** | **+0.1416** | **Direct input component** ($w_C = 0.20$). |
| `glcm_energy` | GLCM Texture | -0.3961 | -0.3842 | Inversely related to disorder/contrast. |
| `circularity` | Derived Geometry | -0.3265 | -0.3308 | Irregular elongated defects have lower circularity. |
| `solidity` | Morphological Convexity| -0.1143 | -0.0840 | Weak negative correlation. |

*Scientific Note: These correlations reflect alignment with the mathematical formula in `index_calculator.py`, not an independently measured physical failure limit.*

---

## 4. Formal Categorization of Claims

### Claims We CAN Honestly Make Today
1. **Dataset Integrity**: The NEU-DET benchmark was parsed, checked, and validated with zero missing files or coordinate errors across 1,770 images and 4,126 defect instances.
2. **Zero-Leakage Splitting**: Train (70%), validation (15%), and test (15%) partitions are strictly isolated at the image level.
3. **Quantitative Feature Extraction**: A comprehensive modular feature extraction pipeline successfully measures 38 geometric, intensity, and GLCM/LBP texture descriptors.
4. **Transparent Index Formulation**: A structured, continuous geometric severity index ($S \in [0, 100]$) can be formulated from measurable 2D planar image characteristics.
5. **Surrogate ML Emulation**: Tree-based gradient boosting models (XGBoost) can reconstruct the constructed piecewise continuous severity index with 97.98% accuracy.

### Claims We CANNOT YET Make
1. ❌ **"The system predicts true physical steel severity with 97.98% accuracy."** (The 97.98% metric reflects surrogate approximation of our own index).
2. ❌ **"The severity categories correspond to ASTM or industrial acceptance standards."** (No international standard defines this 2D pixel equation).
3. ❌ **"The weak segmentation captures exact ground truth defect boundaries."** (Only high-contrast intensity regions are extracted via Otsu filtering).
4. ❌ **"The system measures physical defect depth, volume, or structural failure risk."** (2D planar images without depth calibration cannot yield 3D depth or volumetric loss).
5. ❌ **"The system is robust on Critical defects."** (Due to class imbalance, the model achieves only 20% recall on the Critical class).
6. ❌ **"The full computer-vision detection pipeline has been validated end-to-end."** (The neural YOLO object detector remains to be formally trained and evaluated).

---

## 5. Experiments Required Before Final Claims

1. **Phase 2 YOLO Object Detector Training & Evaluation**:
   - Train YOLOv8n / YOLOv11n on `data/splits/train.txt` and evaluate mAP@0.5, mAP@0.5:0.95, precision, recall, and inference latency on `data/splits/test.txt`.
   - Measure error propagation: test feature extraction and severity estimation using predicted YOLO bounding boxes rather than ground-truth XML boxes.
2. **Class Imbalance & Threshold Recalibration**:
   - Recalibrate severity thresholds using empirical quantile binning (e.g. 25th, 50th, 75th percentiles of index distribution) or apply class-weighted loss / SMOTE to address the 0.46% Critical minority class.
3. **Independent Metallurgical / Expert Grading Study**:
   - Conduct an expert grading rubric survey where steel surface quality inspectors assign ordinal ranks (Low, Medium, High) to a sample of 100 test images to validate the correlation between our algorithmic index and human domain perception.
4. **Class-Specific Severity Calibration**:
   - Formulate class-adapted weights (e.g. higher length weighting for cracks/scratches, higher area weighting for patches and inclusions).
