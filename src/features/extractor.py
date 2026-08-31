import cv2
import numpy as np
from typing import Dict, Any, Tuple
from .geometric import extract_geometric_features
from .intensity import extract_intensity_features
from .texture import extract_texture_features

class DefectFeatureExtractor:
    """
    Unified quantitative feature extractor for steel surface defects.
    Combines geometric, intensity, and texture descriptor engines.
    """
    
    def __init__(self, include_texture: bool = True):
        self.include_texture = include_texture

    def extract(self, image: np.ndarray, mask: np.ndarray, bbox: Tuple[float, float, float, float] = None) -> Dict[str, float]:
        features = {}

        # 1. Geometric Features
        geom_feats = extract_geometric_features(mask=mask, bbox=bbox, image_shape=image.shape)
        features.update(geom_feats)

        # 2. Intensity Features
        inten_feats = extract_intensity_features(image=image, mask=mask)
        features.update(inten_feats)

        # 3. Texture Features
        if self.include_texture:
            tex_feats = extract_texture_features(image=image, mask=mask)
            features.update(tex_feats)

        return features
