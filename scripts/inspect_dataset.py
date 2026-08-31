import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.data.xml_parser import XMLAnnotationParser
from src.data.validator import DatasetValidator
from src.utils.config import save_yaml_config

logger = get_logger("inspect_dataset")

def inspect_dataset(data_dir: str | Path, output_dir: str | Path, config_file: str | Path):
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    annotations_dir = data_path / "ANNOTATIONS"
    if not annotations_dir.exists():
        annotations_dir = data_path / "annotations"
        
    images_dir = data_path / "IMAGES"
    if not images_dir.exists():
        images_dir = data_path / "images"

    if not annotations_dir.exists() or not images_dir.exists():
        logger.error(f"Could not find ANNOTATIONS or IMAGES directory under {data_path}")
        sys.exit(1)

    logger.info(f"Inspecting annotations in {annotations_dir} and images in {images_dir}")

    parser = XMLAnnotationParser(annotations_dir=annotations_dir, images_dir=images_dir)
    records = parser.parse_all()

    validator = DatasetValidator(records=records, images_dir=images_dir)
    val_report = validator.validate()

    # Bounding box statistics dataframe
    bbox_rows = []
    for rec in records:
        if not rec.get("valid", True):
            continue
        for obj in rec.get("valid_objects", []):
            bbox_rows.append({
                "xml_filename": rec["xml_filename"],
                "image_filename": rec["image_filename"],
                "class_name": obj["class_name"],
                "xmin": obj["xmin"],
                "ymin": obj["ymin"],
                "xmax": obj["xmax"],
                "ymax": obj["ymax"],
                "width": obj["bbox_width"],
                "height": obj["bbox_height"],
                "area": obj["bbox_area"],
                "aspect_ratio": obj["bbox_width"] / max(1.0, obj["bbox_height"])
            })

    bbox_df = pd.DataFrame(bbox_rows)

    bbox_stats = {}
    if not bbox_df.empty:
        bbox_stats = {
            "total_boxes": int(len(bbox_df)),
            "width_min": float(bbox_df["width"].min()),
            "width_max": float(bbox_df["width"].max()),
            "width_mean": float(bbox_df["width"].mean()),
            "width_std": float(bbox_df["width"].std()),
            "height_min": float(bbox_df["height"].min()),
            "height_max": float(bbox_df["height"].max()),
            "height_mean": float(bbox_df["height"].mean()),
            "height_std": float(bbox_df["height"].std()),
            "area_min": float(bbox_df["area"].min()),
            "area_max": float(bbox_df["area"].max()),
            "area_mean": float(bbox_df["area"].mean()),
            "area_std": float(bbox_df["area"].std()),
            "aspect_ratio_mean": float(bbox_df["aspect_ratio"].mean())
        }

    dataset_summary = {
        "dataset_directory": str(data_path.resolve()),
        "total_xml_files": val_report["total_annotations"],
        "total_image_files": val_report["total_images_in_dir"],
        "valid_annotations": val_report["valid_annotations_count"],
        "malformed_xml_count": len(val_report["malformed_xml_files"]),
        "missing_images_count": len(val_report["missing_image_files"]),
        "unreferenced_images_count": len(val_report["unreferenced_images"]),
        "invalid_boxes_count": val_report["invalid_boxes_count"],
        "out_of_bound_boxes_count": val_report["out_of_bound_boxes_count"],
        "class_distribution": val_report["class_distribution"],
        "bbox_statistics": bbox_stats
    }

    # Save JSON report
    json_path = out_path / "dataset_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=4)
    logger.info(f"Saved dataset JSON report to {json_path}")

    # Save CSV report of bboxes
    csv_path = out_path / "dataset_report.csv"
    bbox_df.to_csv(csv_path, index=False)
    logger.info(f"Saved dataset CSV report to {csv_path}")

    # Plot Class Distribution
    if val_report["class_distribution"]:
        plt.figure(figsize=(10, 6))
        cls_series = pd.Series(val_report["class_distribution"]).sort_values(ascending=False)
        sns.barplot(x=cls_series.index, y=cls_series.values, palette="crest")
        plt.title("Defect Class Distribution in Dataset", fontsize=14, fontweight="bold")
        plt.xlabel("Defect Category", fontsize=12)
        plt.ylabel("Number of Annotations", fontsize=12)
        plt.xticks(rotation=30)
        plt.tight_layout()
        cls_plot_path = out_path / "class_distribution.png"
        plt.savefig(cls_plot_path, dpi=300)
        plt.close()
        logger.info(f"Saved class distribution plot to {cls_plot_path}")

    # Plot Bounding Box Distribution
    if not bbox_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(bbox_df["area"], kde=True, ax=axes[0], color="teal")
        axes[0].set_title("Bounding Box Area (px²)")
        
        sns.histplot(bbox_df["aspect_ratio"], kde=True, ax=axes[1], color="coral")
        axes[1].set_title("Bounding Box Aspect Ratio (W/H)")
        
        sns.scatterplot(data=bbox_df, x="width", y="height", hue="class_name", ax=axes[2], alpha=0.7)
        axes[2].set_title("Bounding Box Width vs Height")
        
        plt.tight_layout()
        bbox_plot_path = out_path / "bbox_distribution.png"
        plt.savefig(bbox_plot_path, dpi=300)
        plt.close()
        logger.info(f"Saved bounding box distribution plot to {bbox_plot_path}")

    # Generate / Update configs/dataset.yaml
    classes = sorted(list(val_report["class_distribution"].keys()))
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    
    cfg = {
        "dataset_name": "NEU-DET",
        "data_dir": str(data_path.resolve()),
        "images_dir": str(images_dir.resolve()),
        "annotations_dir": str(annotations_dir.resolve()),
        "classes": classes,
        "class_to_id": class_to_id,
        "num_classes": len(classes),
        "image_width": 200,
        "image_height": 200,
        "channels": 1
    }
    save_yaml_config(cfg, config_file)
    logger.info(f"Saved updated dataset YAML configuration to {config_file}")

    print("\n" + "="*70)
    print("DATASET AUDIT SUMMARY")
    print("="*70)
    print(f"Total XML Files        : {val_report['total_annotations']}")
    print(f"Total Image Files      : {val_report['total_images_in_dir']}")
    print(f"Valid Annotations      : {val_report['valid_annotations_count']}")
    print(f"Class Distribution     : {val_report['class_distribution']}")
    print(f"Total Defect Instances : {len(bbox_df)}")
    print(f"Classes Found ({len(classes)})  : {classes}")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect NEU-DET steel defect dataset and generate audit report.")
    parser.add_argument("--data_dir", type=str, default="data/raw/NEU-DET", help="Path to raw dataset directory")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory to save audit reports and figures")
    parser.add_argument("--config_file", type=str, default="configs/dataset.yaml", help="Path to output dataset configuration YAML")
    args = parser.parse_args()

    inspect_dataset(args.data_dir, args.output_dir, args.config_file)
