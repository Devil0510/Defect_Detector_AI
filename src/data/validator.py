import os
from pathlib import Path
from typing import Dict, List, Any, Set
import logging

logger = logging.getLogger("steel_defect.validator")

class DatasetValidator:
    """
    Validates dataset integrity, bounding box coordinates, missing files, and class balances.
    """
    
    def __init__(self, records: List[Dict[str, Any]], images_dir: str | Path):
        self.records = records
        self.images_dir = Path(images_dir)

    def validate(self) -> Dict[str, Any]:
        total_xmls = len(self.records)
        missing_image_files = []
        malformed_xmls = []
        invalid_boxes = []
        out_of_bound_boxes = []
        class_distribution = {}
        valid_records = []
        
        all_image_files = set(self.images_dir.glob("*.jpg")) | set(self.images_dir.glob("*.bmp")) | set(self.images_dir.glob("*.png"))
        image_filenames = {img.name for img in all_image_files}
        referenced_images = set()

        for rec in self.records:
            if not rec.get("valid", True):
                malformed_xmls.append(rec["xml_filename"])
                continue

            img_name = rec["image_filename"]
            referenced_images.add(img_name)
            img_path = self.images_dir / img_name
            
            # Check if image file exists
            if not img_path.exists():
                # Check stem match in case extension differs
                stem_matches = list(self.images_dir.glob(f"{Path(img_name).stem}.*"))
                if stem_matches:
                    img_path = stem_matches[0]
                    rec["image_filename"] = img_path.name
                else:
                    logger.warning(f"Missing image file for annotation {rec['xml_filename']}: {img_name}")
                    missing_image_files.append({"xml": rec["xml_filename"], "expected_image": img_name})

            width = rec["width"]
            height = rec["height"]
            valid_objects = []

            for obj in rec["objects"]:
                cls = obj["class_name"]
                class_distribution[cls] = class_distribution.get(cls, 0) + 1

                xmin, ymin = obj["xmin"], obj["ymin"]
                xmax, ymax = obj["xmax"], obj["ymax"]

                # Check box sanity
                if xmin >= xmax or ymin >= ymax or (xmax - xmin) <= 0 or (ymax - ymin) <= 0:
                    logger.warning(f"Invalid inverted/zero box in {rec['xml_filename']}: [{xmin}, {ymin}, {xmax}, {ymax}]")
                    invalid_boxes.append({
                        "xml": rec["xml_filename"],
                        "class": cls,
                        "bbox": [xmin, ymin, xmax, ymax]
                    })
                    continue

                # Check out of bound boxes
                if width > 0 and height > 0:
                    if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                        logger.warning(f"Box out of bounds in {rec['xml_filename']}: [{xmin}, {ymin}, {xmax}, {ymax}] for img size ({width}x{height})")
                        out_of_bound_boxes.append({
                            "xml": rec["xml_filename"],
                            "class": cls,
                            "bbox": [xmin, ymin, xmax, ymax],
                            "img_size": [width, height]
                        })

                valid_objects.append(obj)

            rec["valid_objects"] = valid_objects
            valid_records.append(rec)

        unreferenced_images = list(image_filenames - referenced_images)

        report = {
            "total_annotations": total_xmls,
            "total_images_in_dir": len(all_image_files),
            "valid_annotations_count": len(valid_records),
            "malformed_xml_files": malformed_xmls,
            "missing_image_files": missing_image_files,
            "unreferenced_images": unreferenced_images,
            "invalid_boxes_count": len(invalid_boxes),
            "invalid_boxes_details": invalid_boxes,
            "out_of_bound_boxes_count": len(out_of_bound_boxes),
            "out_of_bound_boxes_details": out_of_bound_boxes,
            "class_distribution": class_distribution
        }

        return report
