import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from typing import Dict, Any

def extract_texture_features(image: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
    """
    Extracts GLCM texture matrices and Local Binary Pattern (LBP) statistics.
    """
    if len(image.shape) == 3:
        gray = (np.mean(image, axis=2)).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)

    # GLCM matrix computation (quantize to 32 levels for fast matrix computation)
    gray_q = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(gray_q, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)

    contrast = float(np.mean(graycoprops(glcm, 'contrast')))
    correlation = float(np.mean(graycoprops(glcm, 'correlation')))
    energy = float(np.mean(graycoprops(glcm, 'energy')))
    homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))

    # Local Binary Pattern (LBP)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

    if mask is not None and np.sum(mask > 0) > 0:
        lbp_vals = lbp[mask > 0]
    else:
        lbp_vals = lbp.flatten()

    lbp_mean = float(np.mean(lbp_vals))
    lbp_std = float(np.std(lbp_vals))

    return {
        "glcm_contrast": contrast,
        "glcm_correlation": correlation,
        "glcm_energy": energy,
        "glcm_homogeneity": homogeneity,
        "lbp_mean": lbp_mean,
        "lbp_std": lbp_std
    }
