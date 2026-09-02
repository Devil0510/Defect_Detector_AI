import cv2
import numpy as np
from typing import Tuple

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to amplify
    subtle microstructural surface defects such as crazing micro-cracks and rolled-in scale.

    Args:
        image: Input image (grayscale HxW or BGR HxWx3).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        CLAHE-enhanced image in the same format.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 2:
        return clahe.apply(image)
    elif len(image.shape) == 3:
        if image.shape[2] == 1:
            return clahe.apply(image[:, :, 0])[:, :, np.newaxis]
        else:
            # Convert to LAB color space and apply CLAHE to the L (Lightness) channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab_planes = list(cv2.split(lab))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab_merged = cv2.merge(lab_planes)
            return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
    return image
