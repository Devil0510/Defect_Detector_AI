import numpy as np
import cv2
from skimage.measure import regionprops, label
from typing import Dict, Any, Tuple

def extract_geometric_features(mask: np.ndarray, bbox: Tuple[float, float, float, float] = None, image_shape: Tuple[int, int] = (200, 200)) -> Dict[str, float]:
    """
    Extracts 2D geometric and morphological features from a binary defect mask.
    """
    img_h, img_w = image_shape[:2]
    img_area = float(img_h * img_w)

    binary_mask = (mask > 0).astype(np.uint8)
    area_px = float(np.sum(binary_mask))

    if area_px == 0:
        return {
            "defect_area_px": 0.0,
            "bbox_area_px": 0.0,
            "area_ratio": 0.0,
            "bbox_fill_ratio": 0.0,
            "perimeter": 0.0,
            "length_px": 0.0,
            "width_px": 0.0,
            "aspect_ratio": 1.0,
            "circularity": 0.0,
            "compactness": 0.0,
            "eccentricity": 0.0,
            "orientation": 0.0,
            "solidity": 0.0,
            "extent": 0.0,
            "equivalent_diameter": 0.0,
            "connected_components_count": 0.0
        }

    # Contours & Perimeter
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)

    # Bounding Box
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        bw = max(1.0, xmax - xmin)
        bh = max(1.0, ymax - ymin)
    else:
        ys, xs = np.where(binary_mask > 0)
        xmin, xmax = float(np.min(xs)), float(np.max(xs))
        ymin, ymax = float(np.min(ys)), float(np.max(ys))
        bw = max(1.0, xmax - xmin + 1)
        bh = max(1.0, ymax - ymin + 1)

    bbox_area_px = bw * bh
    length_px = max(bw, bh)
    width_px = min(bw, bh)
    aspect_ratio = length_px / max(1.0, width_px)

    area_ratio = area_px / img_area
    bbox_fill_ratio = area_px / max(1.0, bbox_area_px)
    extent = bbox_fill_ratio

    # Circularity & Compactness
    circularity = (4.0 * np.pi * area_px) / max(1.0, perimeter ** 2) if perimeter > 0 else 0.0
    compactness = (perimeter ** 2) / max(1.0, area_px) if area_px > 0 else 0.0
    equivalent_diameter = np.sqrt(4.0 * area_px / np.pi)

    # Regionprops for connected components, eccentricity, solidity
    labeled = label(binary_mask)
    props = regionprops(labeled)
    cc_count = float(len(props))

    eccentricity = 0.0
    orientation = 0.0
    solidity = 0.0

    if props:
        # Take largest component for regionprops metrics
        largest_prop = max(props, key=lambda p: p.area)
        eccentricity = float(largest_prop.eccentricity) if hasattr(largest_prop, "eccentricity") else 0.0
        orientation = float(largest_prop.orientation) if hasattr(largest_prop, "orientation") else 0.0
        solidity = float(largest_prop.solidity) if hasattr(largest_prop, "solidity") else 0.0

    return {
        "defect_area_px": float(area_px),
        "bbox_area_px": float(bbox_area_px),
        "area_ratio": float(area_ratio),
        "bbox_fill_ratio": float(bbox_fill_ratio),
        "perimeter": float(perimeter),
        "length_px": float(length_px),
        "width_px": float(width_px),
        "aspect_ratio": float(aspect_ratio),
        "circularity": float(circularity),
        "compactness": float(compactness),
        "eccentricity": float(eccentricity),
        "orientation": float(orientation),
        "solidity": float(solidity),
        "extent": float(extent),
        "equivalent_diameter": float(equivalent_diameter),
        "connected_components_count": float(cc_count)
    }
