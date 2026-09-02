# Scientific Research & Methodological Audit

This directory contains the comprehensive, rigorous methodological and research integrity audit for the **AI-Based Steel Surface Defect Severity Estimation** project.

---

## Audit Reports Table of Contents

1. [`severity_label_lineage.md`](file:///home/vaseem/PROJECT_DL/reports/research_audit/severity_label_lineage.md)
   - Complete pipeline trace from raw image to evaluation.
   - Analysis of circular surrogate modeling vs independent supervised learning.
   - Diagram of data and label lineage.

2. [`severity_index_audit.md`](file:///home/vaseem/PROJECT_DL/reports/research_audit/severity_index_audit.md)
   - Line-by-line parameter audit of `src/severity/index_calculator.py`.
   - Table of weights, normalizations, and thresholds classified as *Research assumptions*.
   - Metallurgical limitations of linear weighting models.

3. [`standards_audit.md`](file:///home/vaseem/PROJECT_DL/reports/research_audit/standards_audit.md)
   - Audit of industrial specification claims and 2D measurement limitations.
   - Verification that no industrial standard was fabricated or falsely claimed.
   - Explanation of why 2D pixel-based severity formulas are exploratory geometric proxies.

4. [`detection_audit.md`](file:///home/vaseem/PROJECT_DL/reports/research_audit/detection_audit.md)
   - Audit of the object detection stage.
   - Clear finding: `Detection stage incomplete.`
   - Roadmap for training and evaluating YOLOv8/v11 models.

5. [`segmentation_audit.md`](file:///home/vaseem/PROJECT_DL/reports/research_audit/segmentation_audit.md)
   - Mechanics of weak ROI threshold segmentation.
   - Analysis of Otsu contrast bias and failure modes across defect classes.
   - Mandatory reporting terminology (derived weak masks vs ground truth).

6. [`leakage_audit.md`](file:///home/vaseem/PROJECT_DL/reports/research_audit/leakage_audit.md)
   - Partition independence audit across Train (70%), Val (15%), and Test (15%).
   - Verification of zero image-level data leakage and unscaled pipeline artifacts.

7. [`claims_audit.md`](file:///home/vaseem/PROJECT_DL/reports/research_audit/claims_audit.md)
   - Empirical class balance audit (identifying the 0.46% Critical minority class and 20% test recall).
   - Feature-target correlation analysis (Pearson $r$ and Spearman $\rho$).
   - Three mandated claim categories: Claims we CAN make, Claims we CANNOT make, and Required experiments.

8. [`final_recommendation.md`](file:///home/vaseem/PROJECT_DL/reports/research_audit/final_recommendation.md)
   - Project classification: **YELLOW** (*Strong engineering implementation — Significant methodological work required*).
   - 4-step roadmap to advance project to publication-grade **GREEN** status.
