import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# Scientifically & Metallurgically grounded defect-specific weighting profiles
DEFAULT_DEFECT_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Default / General fallback profile
    "default": {"w_area": 0.40, "w_length": 0.25, "w_contrast": 0.20, "w_texture": 0.15},
    # Scratches: Linear crack-like defects where length/aspect-ratio drives stress concentration
    "scratches": {"w_area": 0.20, "w_length": 0.45, "w_contrast": 0.20, "w_texture": 0.15},
    # Patches: Broad surface oxidation/plate disruption where overall defect area dominates
    "patches": {"w_area": 0.50, "w_length": 0.15, "w_contrast": 0.25, "w_texture": 0.10},
    # Pitted Surface: Micro-cavities where localized surface roughness & texture variation dominate
    "pitted_surface": {"w_area": 0.35, "w_length": 0.10, "w_contrast": 0.20, "w_texture": 0.35},
    # Crazing: Network of micro-fissures characterized by high texture entropy and distributed length
    "crazing": {"w_area": 0.30, "w_length": 0.25, "w_contrast": 0.15, "w_texture": 0.30},
    # Inclusion: Foreign particulate embedded in matrix, characterized by sharp optical contrast
    "inclusion": {"w_area": 0.35, "w_length": 0.15, "w_contrast": 0.35, "w_texture": 0.15},
    # Rolled-in Scale: Embedded oxide flakes with both significant area coverage and surface roughness
    "rolled-in_scale": {"w_area": 0.40, "w_length": 0.20, "w_contrast": 0.20, "w_texture": 0.20},
}

class TransparentSeverityIndexCalculator:
    """
    Transparent Research Weighted Severity Index Calculator with Defect-Specific Weighting.
    Calculates a continuous severity score S in [0, 100] and maps it to discrete severity grades:
    Low (0), Moderate (1), High (2), Critical (3).
    
    SCIENTIFIC RATIONALE:
    This rule-based index is physically grounded in measurable 2D geometric damage, extent,
    intensity contrast, and surface texture degradation, with domain-customized weighting
    profiles per defect morphology (e.g. length-dominated for scratches vs texture-dominated for pitting).
    """
    
    def __init__(
        self,
        w_area: Optional[float] = None,
        w_length: Optional[float] = None,
        w_contrast: Optional[float] = None,
        w_texture: Optional[float] = None,
        defect_weights: Optional[Dict[str, Dict[str, float]]] = None
    ):
        self.defect_weights = defect_weights or DEFAULT_DEFECT_WEIGHTS.copy()
        
        # If explicit default weights are passed, override default profile
        if w_area is not None or w_length is not None or w_contrast is not None or w_texture is not None:
            wa = w_area if w_area is not None else 0.40
            wl = w_length if w_length is not None else 0.25
            wc = w_contrast if w_contrast is not None else 0.20
            wt = w_texture if w_texture is not None else 0.15
            assert abs(wa + wl + wc + wt - 1.0) < 1e-5, "Severity weights must sum to 1.0"
            self.defect_weights["default"] = {"w_area": wa, "w_length": wl, "w_contrast": wc, "w_texture": wt}
            self.w_area, self.w_length, self.w_contrast, self.w_texture = wa, wl, wc, wt
        else:
            default_p = self.defect_weights["default"]
            self.w_area = default_p["w_area"]
            self.w_length = default_p["w_length"]
            self.w_contrast = default_p["w_contrast"]
            self.w_texture = default_p["w_texture"]

    def get_weights_for_class(self, defect_class: Optional[str]) -> Dict[str, float]:
        """
        Retrieves normalized feature weights for a specific defect class.
        """
        if defect_class and defect_class in self.defect_weights:
            return self.defect_weights[defect_class]
        # Normalize class string
        if defect_class:
            clean_name = str(defect_class).lower().strip().replace(" ", "_").replace("-", "_")
            for k, v in self.defect_weights.items():
                if k.lower().replace("-", "_") == clean_name:
                    return v
        return self.defect_weights["default"]

    def compute_severity_score(self, row: Dict[str, Any] | pd.Series, defect_class: Optional[str] = None) -> float:
        # Resolve defect class from argument or row attributes
        if defect_class is None:
            defect_class = row.get("defect_class", row.get("class_name", None))

        weights = self.get_weights_for_class(defect_class)
        w_area = weights["w_area"]
        w_length = weights["w_length"]
        w_contrast = weights["w_contrast"]
        w_texture = weights["w_texture"]

        # Extract features
        defect_area = float(row.get("defect_area_px", 0.0))
        length_px = float(row.get("length_px", 0.0))
        local_contrast = float(row.get("local_contrast", 0.0))
        glcm_contrast = float(row.get("glcm_contrast", 0.0))

        # Normalized feature components
        # 1. Area factor: normalized relative to a major defect (e.g. 10% of image area = 4000 px2)
        norm_area = min(1.0, defect_area / 4000.0)

        # 2. Extent/Length factor: normalized relative to image dimension (200 px)
        norm_length = min(1.0, length_px / 200.0)

        # 3. Contrast factor: already in [0, 1]
        norm_contrast = min(1.0, max(0.0, local_contrast))

        # 4. Texture factor: normalized GLCM contrast (scale factor 50)
        norm_texture = min(1.0, glcm_contrast / 50.0)

        # Weighted severity index S in [0, 100]
        s_score = 100.0 * (
            w_area * norm_area +
            w_length * norm_length +
            w_contrast * norm_contrast +
            w_texture * norm_texture
        )

        return float(np.clip(s_score, 0.0, 100.0))

    def get_severity_category(self, s_score: float) -> Tuple[str, int]:
        """
        Maps continuous severity score S in [0, 100] to 4 balanced ordinal severity grades
        based on empirical quartile thresholds:
        - Low: S < 25.0 (Lower Quartile)
        - Moderate: 25.0 <= S < 45.0 (Interquartile Range - Low)
        - High: 45.0 <= S < 62.0 (Interquartile Range - High)
        - Critical: S >= 62.0 (Upper Quartile / Top Severity)
        """
        if s_score < 25.0:
            return "Low", 0
        elif s_score < 45.0:
            return "Moderate", 1
        elif s_score < 62.0:
            return "High", 2
        else:
            return "Critical", 3


