import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO
import logging
from src.data.clahe import apply_clahe

logger = logging.getLogger("steel_defect.yolo_adapter")

class YOLODefectDetector:
    """
    Adapter for trained YOLO object detection models in steel defect pipeline.
    Supports optional CLAHE contrast amplification for subtle defect detection.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.5
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO detector weights not found at: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit

    def detect(self, image_path_or_array: str | Path | np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on image and returns detected defects.
        """
        inp = image_path_or_array
        if self.use_clahe:
            if isinstance(inp, (str, Path)):
                img = cv2.imread(str(inp))
                if img is not None:
                    inp = apply_clahe(img, clip_limit=self.clahe_clip_limit)
            elif isinstance(inp, np.ndarray):
                inp = apply_clahe(inp, clip_limit=self.clahe_clip_limit)

        results = self.model.predict(
            inp,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "bbox": (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name
                })

        return detections

