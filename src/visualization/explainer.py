import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple

class DefectSeverityExplainer:
    """
    Advanced visual explanation engine for steel surface defect severity predictions.
    Generates multi-panel diagnostic visualizations displaying localized defect,
    binary mask contour, Grad-CAM severity heatmap, quantitative metrics,
    neural predictive severity with MC Dropout uncertainty intervals (mean ± 1.96*sigma),
    and quality control triage indicators.
    """

    @staticmethod
    def create_explanation_plot(
        image: np.ndarray,
        bbox: Tuple[float, float, float, float],
        mask: np.ndarray,
        defect_class: str,
        features: Dict[str, float],
        severity_score: float,
        severity_category: str,
        heatmap: np.ndarray = None,
        uncertainty_info: Dict[str, Any] = None,
        output_path: str | Path = "explanation.png"
    ):
        n_panels = 4 if heatmap is not None else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

        # 1. Image with Bounding Box
        img_bbox = image.copy()
        xmin, ymin, xmax, ymax = [int(round(v)) for v in bbox]
        cv2.rectangle(img_bbox, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(img_bbox, f"{defect_class}", (xmin, max(15, ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        axes[0].imshow(cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB) if len(img_bbox.shape) == 3 else img_bbox, cmap="gray")
        axes[0].set_title(f"1. Localization: {defect_class}", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # 2. Derived Segmentation Mask Contour Overlay
        img_contour = image.copy()
        if len(img_contour.shape) == 2:
            img_contour = cv2.cvtColor(img_contour, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 2)

        axes[1].imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
        axes[1].set_title("2. Derived Weak Mask", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        panel_idx = 2

        # 3. Grad-CAM Spatial Severity Heatmap (if provided)
        if heatmap is not None:
            # Crop ROI heatmap and overlay
            h_img, w_img = image.shape[:2]
            heat_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
            color_heat = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
            overlay_full = cv2.addWeighted(img_contour, 0.6, color_heat, 0.4, 0)

            axes[2].imshow(cv2.cvtColor(overlay_full, cv2.COLOR_BGR2RGB))
            axes[2].set_title("3. Grad-CAM Severity Heatmap", fontsize=12, fontweight="bold")
            axes[2].axis("off")
            panel_idx = 3

        # 4. Quantitative Measurements & Severity Badge Panel
        axes[panel_idx].axis("off")
        color_map = {"Low": "green", "Moderate": "blue", "High": "orange", "Critical": "red"}
        badge_color = color_map.get(severity_category, "black")

        unc_text = ""
        triage_text = ""
        if uncertainty_info:
            mean_s = uncertainty_info.get("mean_severity_score", severity_score)
            std_s = uncertainty_info.get("std_severity_score", 0.0)
            ci_l = uncertainty_info.get("ci_95_lower", max(0.0, mean_s - 1.96*std_s))
            ci_u = uncertainty_info.get("ci_95_upper", min(100.0, mean_s + 1.96*std_s))
            triage = uncertainty_info.get("requires_human_triage", False)

            unc_text = (
                f"── UNCERTAINTY ESTIMATION (MC Dropout) ──\n"
                f"Neural Score (μ)  : {mean_s:.1f} ± {std_s:.2f} pts\n"
                f"95% Credible Int. : [{ci_l:.1f}, {ci_u:.1f}]\n"
            )
            triage_text = "[QC TRIAGE: HUMAN INSPECTION REQUIRED]" if triage else "[QC STATUS: HIGH CERTAINTY]"

        summary_text = (
            f"QUANTITATIVE DEFECT METRICS\n"
            f"─────────────────────────────────────\n"
            f"Defect Area       : {features.get('defect_area_px', 0):.1f} px² ({features.get('area_ratio', 0)*100:.2f}%)\n"
            f"Aspect Ratio      : {features.get('aspect_ratio', 1.0):.2f}\n"
            f"Circularity       : {features.get('circularity', 0):.3f}\n"
            f"Local Contrast    : {features.get('local_contrast', 0):.3f}\n"
            f"GLCM Contrast     : {features.get('glcm_contrast', 0):.2f}\n"
            f"─────────────────────────────────────\n"
            f"SEVERITY INDEX    : {severity_score:.1f} / 100\n"
            f"SEVERITY GRADE    : {severity_category.upper()}\n"
            f"{unc_text}"
        )

        axes[panel_idx].text(0.05, 0.95, summary_text, transform=axes[panel_idx].transAxes, fontsize=10, family="monospace", verticalalignment="top")
        axes[panel_idx].text(0.05, 0.12, f"GRADE: {severity_category.upper()}", transform=axes[panel_idx].transAxes, fontsize=15, fontweight="bold", color=badge_color)
        if triage_text:
            t_color = "red" if "REQUIRED" in triage_text else "green"
            axes[panel_idx].text(0.05, 0.04, triage_text, transform=axes[panel_idx].transAxes, fontsize=11, fontweight="bold", color=t_color)

        plt.tight_layout()
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300)
        plt.close()
