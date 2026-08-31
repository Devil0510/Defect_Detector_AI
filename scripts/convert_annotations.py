import os
import sys
import argparse
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_yaml_config
from src.data.xml_parser import XMLAnnotationParser
from src.data.validator import DatasetValidator

logger = get_logger("convert_annotations")

def convert_to_yolo(data_dir: str | Path, output_dir: str | Path, config_file: str | Path):
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml_config(config_file)
    class_to_id = cfg.get("class_to_id", {})

    annotations_dir = data_path / "ANNOTATIONS"
    if not annotations_dir.exists():
        annotations_dir = data_path / "annotations"
        
    images_dir = data_path / "IMAGES"
    if not images_dir.exists():
        images_dir = data_path / "images"

    logger.info(f"Parsing annotations from {annotations_dir} for YOLO format conversion...")
    parser = XMLAnnotationParser(annotations_dir=annotations_dir, images_dir=images_dir)
    records = parser.parse_all()

    validator = DatasetValidator(records=records, images_dir=images_dir)
    val_report = validator.validate()

    total_labels_written = 0
    total_boxes_converted = 0

    for rec in records:
        if not rec.get("valid", True):
            continue

        xml_stem = Path(rec["xml_filename"]).stem
        label_file = out_path / f"{xml_stem}.txt"

        img_w = rec["width"] if rec["width"] > 0 else 200
        img_h = rec["height"] if rec["height"] > 0 else 200

        lines = []
        for obj in rec.get("valid_objects", []):
            cls_name = obj["class_name"]
            if cls_name not in class_to_id:
                logger.warning(f"Unknown class '{cls_name}' in {rec['xml_filename']}, skipping.")
                continue

            cls_id = class_to_id[cls_name]

            xmin, ymin = obj["xmin"], obj["ymin"]
            xmax, ymax = obj["xmax"], obj["ymax"]

            # Clamp coordinates to image boundaries
            xmin = max(0.0, min(float(img_w), xmin))
            ymin = max(0.0, min(float(img_h), ymin))
            xmax = max(0.0, min(float(img_w), xmax))
            ymax = max(0.0, min(float(img_h), ymax))

            bw = xmax - xmin
            bh = ymax - ymin

            if bw <= 0 or bh <= 0:
                continue

            # Normalized YOLO coordinates
            x_center = (xmin + xmax) / (2.0 * img_w)
            y_center = (ymin + ymax) / (2.0 * img_h)
            norm_w = bw / img_w
            norm_h = bh / img_h

            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            total_boxes_converted += 1

        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        total_labels_written += 1

    print("\n" + "="*60)
    print("YOLO ANNOTATION CONVERSION COMPLETE")
    print("="*60)
    print(f"Label files written : {total_labels_written}")
    print(f"Boxes converted     : {total_boxes_converted}")
    print(f"Output directory    : {out_path.resolve()}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML annotations to YOLO format.")
    parser.add_argument("--data_dir", type=str, default="data/raw/NEU-DET", help="Path to raw dataset directory")
    parser.add_argument("--output_dir", type=str, default="data/interim/labels", help="Output directory for YOLO label files")
    parser.add_argument("--config_file", type=str, default="configs/dataset.yaml", help="Dataset configuration file")
    args = parser.parse_args()

    convert_to_yolo(args.data_dir, args.output_dir, args.config_file)
