# Severity Label Lineage & Circularity Audit

## 1. Pipeline Lineage Trace

The complete data and target generation path currently implemented in the codebase is traced below:

```text
┌────────────────────────────────────────────────────────┐
│ Raw 2D Surface Image (200x200 px) & Pascal VOC XML BBox│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Weak ROI Threshold Segmentation (Adaptive/Otsu Filter) │
└───────────────────────────┬────────────────────────────┘
                            │ (Binary Mask)
                            ▼
┌────────────────────────────────────────────────────────┐
│ Feature Extraction Engine (38 Descriptors)             │
│ - Geometric: defect_area_px, length_px, aspect_ratio...│
│ - Intensity: local_contrast, mean, std...              │
│ - Texture: glcm_contrast, energy, LBP...               │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │ (Same Extracted Features)
              ▼                           ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ Severity Index Calculator │   │ Feature Table (X)         │
│ S = 100*(0.4A + 0.25L +   │   │ (30 Feature Columns)      │
│          0.2C + 0.15T)    │   └─────────────┬─────────────┘
└─────────────┬─────────────┘                 │
              │ (Continuous Score S)          │
              ▼                               │
┌───────────────────────────┐                 │
│ Severity Binning [0..3]   │                 │
│ Low <25 <= Mod <50 <=     │                 │
│ High <75 <= Critical      │                 │
└─────────────┬─────────────┘                 │
              │ (Target Label y)              │
              ▼                               ▼
┌───────────────────────────────────────────────────────────┐
│ Supervised ML Training (Random Forest / XGBoost)          │
│ Fits X (Features) -> y (Target generated from features)   │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│ Evaluation on Held-Out Test Set (644 Defect Instances)    │
│ Target y_test generated using the exact same S formula!   │
│ Result: 97.98% Accuracy (Surrogate Reconstruction)        │
└───────────────────────────────────────────────────────────┘
```

---

## 2. Lineage Audit Inquiries

### 1. How are severity scores generated?
Severity scores $S \in [0, 100]$ are generated analytically by `TransparentSeverityIndexCalculator.compute_severity_score()` using a deterministic linear combination of 4 normalized feature components:
$$S = 100 \times \left( 0.40 \cdot \hat{A} + 0.25 \cdot \hat{L} + 0.20 \cdot \hat{C} + 0.15 \cdot \hat{T} \right)$$
where $\hat{A} = \min(1.0, \text{defect\_area\_px} / 4000)$, $\hat{L} = \min(1.0, \text{length\_px} / 200)$, $\hat{C} = \text{local\_contrast}$, and $\hat{T} = \min(1.0, \text{glcm\_contrast} / 50)$.

### 2. Which features are used to generate severity?
The formula directly consumes:
- `defect_area_px` (Derived weak mask pixel count)
- `length_px` (Bounding box / defect maximum dimension)
- `local_contrast` (Intensity difference between defect and background)
- `glcm_contrast` (Gray-Level Co-occurrence Matrix texture contrast)

### 3. How is continuous severity converted into Low / Moderate / High / Critical?
Arbitrary equal interval thresholds are applied:
- **Low**: $S < 25.0$ (Grade 0)
- **Moderate**: $25.0 \le S < 50.0$ (Grade 1)
- **High**: $50.0 \le S < 75.0$ (Grade 2)
- **Critical**: $S \ge 75.0$ (Grade 3)

### 4. Are the same features provided to the ML models?
**YES.** The ML feature matrix $X$ provided to XGBoost and Random Forest contains all 4 features directly used in the target formulation (`defect_area_px`, `length_px`, `local_contrast`, `glcm_contrast`) along with 26 collinear/derived metrics (`bbox_area_px`, `area_ratio`, `perimeter`, `width_px`, `aspect_ratio`, etc.).

### 5. Are the test-set labels generated using the same formula?
**YES.** The test set labels ($y_{\text{test}}$) are generated using the exact same deterministic index function and threshold boundaries.

### 6. Do any independent human / expert labels exist in the dataset?
**NO.** The NEU-DET dataset contains only defect bounding boxes and categorical defect class names (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`). No human metallurgist, quality control inspector, or domain expert assigned severity scores or criticality grades to these images.

### 7. Does any external standard actually define the implemented severity formula?
**NO.** Neither ASTM, ISO, nor EN standards define this mathematical weighting equation. It is a heuristic formula devised within this research implementation.

---

## 3. Circularity Determination

### Scientific Conclusion:
> **The current experiment is NOT independent supervised learning of physical defect severity.**
> **It is classified as:**
> **Category B + C: Surrogate Modeling of a Deterministic Severity Index & Pseudo-label Classification.**

The reported **97.98% test accuracy** and **0.0202 MAE** demonstrate that an XGBoost decision tree ensemble can successfully learn the piecewise linear and thresholded mathematical function programmed into `index_calculator.py` when given the exact inputs to that function.

It does **not** demonstrate that the system can predict actual physical steel defect severity in an industrial or metallurgical sense without external validation against physical failure criteria or human expert ground truth.
