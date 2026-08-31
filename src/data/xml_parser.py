import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger("steel_defect.xml_parser")

class XMLAnnotationParser:
    """
    Parser for Pascal VOC formatted XML annotations in steel surface defect datasets.
    """
    
    def __init__(self, annotations_dir: str | Path, images_dir: str | Path):
        self.annotations_dir = Path(annotations_dir)
        self.images_dir = Path(images_dir)

    def parse_single_file(self, xml_path: Path) -> Dict[str, Any]:
        """
        Parses a single XML file and validates XML syntax and elements.
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.warning(f"Malformed XML syntax in file {xml_path.name}: {e}")
            return {"filename": xml_path.name, "valid": False, "error": f"Malformed XML syntax: {e}", "objects": []}

        filename_elem = root.find("filename")
        filename = filename_elem.text.strip() if filename_elem is not None and filename_elem.text else xml_path.stem + ".jpg"

        size_elem = root.find("size")
        width, height, depth = 0, 0, 1
        if size_elem is not None:
            w_elem = size_elem.find("width")
            h_elem = size_elem.find("height")
            d_elem = size_elem.find("depth")
            if w_elem is not None and w_elem.text:
                width = int(float(w_elem.text))
            if h_elem is not None and h_elem.text:
                height = int(float(h_elem.text))
            if d_elem is not None and d_elem.text:
                depth = int(float(d_elem.text))

        objects = []
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None or not name_elem.text:
                logger.warning(f"Missing object name tag in {xml_path.name}")
                continue
            class_name = name_elem.text.strip()

            bndbox = obj.find("bndbox")
            if bndbox is None:
                logger.warning(f"Missing bndbox tag in object in {xml_path.name}")
                continue

            try:
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
            except (AttributeError, ValueError, TypeError) as err:
                logger.warning(f"Invalid coordinate format in {xml_path.name}: {err}")
                continue

            objects.append({
                "class_name": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "bbox_width": xmax - xmin,
                "bbox_height": ymax - ymin,
                "bbox_area": (xmax - xmin) * (ymax - ymin)
            })

        return {
            "xml_filename": xml_path.name,
            "image_filename": filename,
            "width": width,
            "height": height,
            "depth": depth,
            "objects": objects,
            "valid": True
        }

    def parse_all(self) -> List[Dict[str, Any]]:
        """
        Parses all XML annotation files in annotations_dir.
        """
        xml_files = sorted(list(self.annotations_dir.glob("*.xml")))
        logger.info(f"Found {len(xml_files)} XML files in {self.annotations_dir}")
        records = []
        for xml_file in xml_files:
            record = self.parse_single_file(xml_file)
            records.append(record)
        return records
