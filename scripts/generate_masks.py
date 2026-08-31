import os
import sys
import cv2
import argparse
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.data.xml_parser import XMLAnnotationParser
from src.data.validator import DatasetValidator
from src.segmentation.weak_segmenter import WeakROIThresholdSegmenter

logger = get_logger("generate_masks")

def main(data_dir: str | Path, output_dir: str | Path):
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    annotations_dir = data_path / "ANNOTATIONS"
    if not annotations_dir.exists():
        annotations_dir = data_path / "annotations"
        
    images_dir = data_path / "IMAGES"
    if not images_dir.exists():
        images_dir = data_path / "images"

    logger.info(f"Parsing annotations from {annotations_dir} for weak ROI mask generation...")
    parser = XMLAnnotationParser(annotations_dir=annotations_dir, images_dir=images_dir)
    records = parser.parse_all()

    validator = DatasetValidator(records=records, images_dir=images_dir)
    validator.validate()

    segmenter = WeakROIThresholdSegmenter()
    masks_generated = 0

    for rec in records:
        if not rec.get("valid", True):
            continue

        img_name = rec["image_filename"]
        img_path = images_dir / img_name
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        composite_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for obj in rec.get("valid_objects", []):
            bbox = (obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"])
            obj_mask = segmenter.segment_roi(image, bbox)
            composite_mask = np.maximum(composite_mask, obj_mask)

        out_mask_path = out_path / f"{Path(img_name).stem}_mask.png"
        cv2.imwrite(str(out_mask_path), composite_mask * 255)
        masks_generated += 1

    print("\n" + "="*60)
    print("WEAK DERIVED SEGMENTATION MASK GENERATION COMPLETE")
    print("="*60)
    print(f"Masks generated : {masks_generated}")
    print(f"Saved to        : {out_path.resolve()}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binary ROI weak segmentation masks.")
    parser.add_argument("--data_dir", type=str, default="data/raw/NEU-DET", help="Path to raw dataset directory")
    parser.add_argument("--output_dir", type=str, default="data/interim/masks", help="Output directory for binary mask PNG files")
    args = parser.parse_args()

    main(args.data_dir, args.output_dir)
