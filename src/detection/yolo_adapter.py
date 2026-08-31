import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO
import logging

logger = logging.getLogger("steel_defect.yolo_adapter")

class YOLODefectDetector:
    """
    Adapter for trained YOLO object detection models in steel defect pipeline.
    """

    def __init__(self, model_path: str | Path, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO detector weights not found at: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image_path_or_array: str | Path | np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on image and returns detected defects.
        """
        results = self.model.predict(
            image_path_or_array,
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
