# Weak Derived Segmentation Stage Audit

## 1. Mechanics of `src/segmentation/weak_segmenter.py`

The weak segmentation module produces binary masks ($0 = \text{background}, 1 = \text{defect}$) through classical computer vision image processing within localized Region-of-Interest (ROI) bounding boxes:

1. **ROI Extraction**: Clamps bounding box coordinates $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ to image boundaries $(200 \times 200)$ and crops the ROI sub-image.
2. **Noise Smoothing**: Applies Gaussian smoothing filter ($5 \times 5$ kernel) to attenuate high-frequency sensor noise.
3. **Contrast-Adaptive Binarization**: Compares the mean pixel intensity of the ROI ($\mu_{\text{roi}}$) against the global image mean ($\mu_{\text{bg}}$).
   - If $\mu_{\text{roi}} < \mu_{\text{bg}}$ (darker defect): Inverts the blurred ROI and executes Otsu automatic thresholding.
   - If $\mu_{\text{roi}} \ge \mu_{\text{bg}}$ (brighter defect): Executes standard Otsu automatic thresholding directly.
4. **Morphological Filtering**: Applies morphological **Closing** ($3 \times 3$ rectangular structuring element) to bridge micro-discontinuities followed by **Opening** to remove isolated single-pixel noise.
5. **Spatial Placement**: Inserts the segmented binary ROI back into a full-resolution ($200 \times 200$) zero-initialized binary image array.

---

## 2. Critical Scientific Limitations

### A. Lack of Pixel-Level Ground Truth
- The NEU-DET dataset provides **only bounding box annotations** ($[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$), **not polygon or pixel segmentation masks**.
- Therefore, **true segmentation metrics (e.g., Intersection-over-Union (mIoU), Dice Similarity Coefficient (DSC), Pixel Accuracy) CANNOT be quantitatively computed** against human ground truth.
- These masks must strictly be documented as **"derived weak masks"** or **"algorithmically generated pseudo-masks"**.

### B. High-Contrast Bias vs True Defect Boundaries
- Otsu thresholding identifies bimodality in the intensity histogram within the box.
- Consequently, it selects **regions of maximum local intensity contrast**, which may only capture the darkest core of an inclusion or pit while ignoring faint transitional edge boundaries.

### C. Class-Specific Segmentation Discrepancies
The uniform Otsu + morphological filter behaves differently across the 6 defect typologies:

| Defect Class | Morphological Nature | Segmentation Efficacy / Failure Mode |
| :--- | :--- | :--- |
| **Patches (`patches`)** | Large, contiguous dark surface regions. | **High**: Cleanly segments large connected plateaus. |
| **Inclusion (`inclusion`)** | Compact, isolated dark particles. | **Moderate**: Separates high-contrast centers well. |
| **Pitted Surface (`pitted_surface`)** | Granular multi-point cavities. | **Mixed**: Undersegments subtle micro-pits; merges nearby pits. |
| **Rolled-in Scale (`rolled-in_scale`)** | Textured, scaly streaks of varying gray levels. | **Moderate**: Extracts coarse textures, misses low-contrast scale. |
| **Scratches (`scratches`)** | Thin, high aspect-ratio lines. | **Moderate/Poor**: Disconnects thin lines if contrast varies along the scratch path. |
| **Crazing (`crazing`)** | Intricate web of micro-cracks. | **Poor**: Morphological closing fills internal gaps between micro-cracks, inflating `defect_area_px`. |

---

## 3. Mandatory Reporting Terminology

In all research documentation, artifacts, and figures:
- **Prohibited Term**: *"Ground truth defect mask"*
- **Mandatory Terms**: *"Algorithmically derived mask"*, *"Weak ROI threshold mask"*, *"Derived binary pseudo-mask"*.
