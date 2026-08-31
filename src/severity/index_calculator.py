import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

class TransparentSeverityIndexCalculator:
    """
    Transparent Research Weighted Severity Index Calculator.
    Calculates a continuous severity score S in [0, 100] and maps it to discrete severity grades:
    Low (0), Moderate (1), High (2), Critical (3).
    
    SCIENTIFIC RATIONALE:
    This rule-based index is physically grounded in measurable 2D geometric damage, extent,
    intensity contrast, and surface texture degradation.
    """
    
    def __init__(self, w_area: float = 0.40, w_length: float = 0.25, w_contrast: float = 0.20, w_texture: float = 0.15):
        assert abs(w_area + w_length + w_contrast + w_texture - 1.0) < 1e-5, "Severity weights must sum to 1.0"
        self.w_area = w_area
        self.w_length = w_length
        self.w_contrast = w_contrast
        self.w_texture = w_texture

    def compute_severity_score(self, row: Dict[str, Any] | pd.Series) -> float:
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
            self.w_area * norm_area +
            self.w_length * norm_length +
            self.w_contrast * norm_contrast +
            self.w_texture * norm_texture
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

