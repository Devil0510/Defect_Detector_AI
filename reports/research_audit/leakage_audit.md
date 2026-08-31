# Data Leakage & Split Independence Audit

## 1. Image-Level Independence Verification

The dataset splitting logic implemented in `src/data/splitter.py` was audited for cross-partition overlap:

| Partition | Total Images | Total Defect Instances | Shared with Train | Shared with Val | Shared with Test | Overlap Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Set** | 1,237 (70%) | 2,888 | — | 0 images (0%) | 0 images (0%) | **Zero Leakage** |
| **Validation Set** | 264 (15%) | 594 | 0 images (0%) | — | 0 images (0%) | **Zero Leakage** |
| **Held-Out Test Set**| 269 (15%) | 644 | 0 images (0%) | 0 images (0%) | — | **Zero Leakage** |

---

## 2. Derived Artifacts & Pipeline Component Leakage Audit

| Component | Methodology Checked | Risk of Information Leakage | Audit Result |
| :--- | :--- | :--- | :---: |
| **Weak Mask Generation** | Independent per-image ROI computation. | None. Processing is local to single images. | **Clean** |
| **Feature Extraction** | Extracted independently per defect bounding box without global dataset statistics. | None. No dataset-wide centering, PCA, or global scaling applied. | **Clean** |
| **Severity Index Formulation** | Fixed deterministic formula applied sample-by-sample. | None. Formula parameters were not optimized on test distributions. | **Clean** |
| **ML Model Training** | Trained strictly on `X_train` (2,888 instances) from `data/splits/train.txt`. | None. `X_test` (644 instances) was held out until evaluation. | **Clean** |
| **Model Ensembles (RF / XGBoost)**| Tree-based decision boundaries invariant to monotonic feature transformations. | No data-dependent scalers (e.g. `StandardScaler`) fitted across splits. | **Clean** |

---

## 3. Residual Dataset Characteristics & Nuances
- **Strip Batch Grouping**: The raw NEU-DET benchmark was captured in sets of 300 images per class. Because strip batch/heat IDs are not annotated in the original Northeastern University dataset, splitting was performed via stratified image shuffling with a fixed random seed (`42`).
- **Defect Multiplicity**: When an image contains multiple defect bounding boxes (e.g., multiple pitted pinholes or scratch lines), all defect instances from that image are confined to the **same split partition**, strictly preventing cross-split image leakage.
