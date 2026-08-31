import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict, Any

def extract_intensity_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Extracts statistical intensity moments and local contrast characteristics of the defect region.
    """
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    defect_pixels = gray[mask > 0]
    background_pixels = gray[mask == 0]

    if len(defect_pixels) == 0:
        return {
            "mean_intensity": 0.0,
            "std_intensity": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "min_intensity": 0.0,
            "max_intensity": 0.0,
            "intensity_range": 0.0,
            "local_contrast": 0.0
        }

    mean_val = float(np.mean(defect_pixels))
    std_val = float(np.std(defect_pixels))
    min_val = float(np.min(defect_pixels))
    max_val = float(np.max(defect_pixels))
    range_val = max_val - min_val

    skew_val = float(skew(defect_pixels)) if len(defect_pixels) > 2 else 0.0
    kurt_val = float(kurtosis(defect_pixels)) if len(defect_pixels) > 2 else 0.0

    bg_mean = float(np.mean(background_pixels)) if len(background_pixels) > 0 else 128.0
    # Michelson / RMS Local Contrast
    local_contrast = abs(mean_val - bg_mean) / max(1.0, mean_val + bg_mean)

    return {
        "mean_intensity": mean_val,
        "std_intensity": std_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "min_intensity": min_val,
        "max_intensity": max_val,
        "intensity_range": range_val,
        "local_contrast": float(local_contrast)
    }
