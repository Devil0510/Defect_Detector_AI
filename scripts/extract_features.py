import os
import sys
import cv2
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.data.xml_parser import XMLAnnotationParser
from src.data.validator import DatasetValidator
from src.segmentation.weak_segmenter import WeakROIThresholdSegmenter
from src.features.extractor import DefectFeatureExtractor

logger = get_logger("extract_features")

def main(data_dir: str | Path, output_file: str | Path):
    data_path = Path(data_dir)
    out_file = Path(output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    annotations_dir = data_path / "ANNOTATIONS"
    if not annotations_dir.exists():
        annotations_dir = data_path / "annotations"
        
    images_dir = data_path / "IMAGES"
    if not images_dir.exists():
        images_dir = data_path / "images"

    logger.info(f"Parsing annotations from {annotations_dir} for quantitative feature extraction...")
    parser = XMLAnnotationParser(annotations_dir=annotations_dir, images_dir=images_dir)
    records = parser.parse_all()

    validator = DatasetValidator(records=records, images_dir=images_dir)
    validator.validate()

    segmenter = WeakROIThresholdSegmenter()
    extractor = DefectFeatureExtractor(include_texture=True)

    extracted_rows = []

    for rec in tqdm(records, desc="Extracting Defect Features"):
        if not rec.get("valid", True):
            continue

        img_name = rec["image_filename"]
        img_path = images_dir / img_name
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        for obj_idx, obj in enumerate(rec.get("valid_objects", [])):
            bbox = (obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"])
            mask = segmenter.segment_roi(image, bbox)

            feats = extractor.extract(image, mask, bbox=bbox)
            
            row = {
                "xml_filename": rec["xml_filename"],
                "image_filename": img_name,
                "defect_index": obj_idx,
                "defect_class": obj["class_name"],
                "bbox_xmin": obj["xmin"],
                "bbox_ymin": obj["ymin"],
                "bbox_xmax": obj["xmax"],
                "bbox_ymax": obj["ymax"],
            }
            row.update(feats)
            extracted_rows.append(row)

    df = pd.DataFrame(extracted_rows)
    df.to_csv(out_file, index=False)

    print("\n" + "="*70)
    print("QUANTITATIVE DEFECT FEATURE EXTRACTION COMPLETE")
    print("="*70)
    print(f"Total defects extracted : {len(df)}")
    print(f"Features extracted ({len(df.columns)}) : {list(df.columns)}")
    print(f"Saved dataset to        : {out_file.resolve()}")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract quantitative geometric, intensity, and texture features.")
    parser.add_argument("--data_dir", type=str, default="data/raw/NEU-DET", help="Path to raw dataset directory")
    parser.add_argument("--output_file", type=str, default="data/processed/defect_features.csv", help="CSV path for extracted feature dataset")
    args = parser.parse_args()

    main(args.data_dir, args.output_file)
