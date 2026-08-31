import os
import sys
import argparse
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.data.xml_parser import XMLAnnotationParser
from src.data.validator import DatasetValidator
from src.data.splitter import StratifiedDatasetSplitter

logger = get_logger("create_splits")

def main(data_dir: str | Path, output_dir: str | Path, seed: int):
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

    logger.info(f"Parsing annotations from {annotations_dir} for split creation...")
    parser = XMLAnnotationParser(annotations_dir=annotations_dir, images_dir=images_dir)
    records = parser.parse_all()

    validator = DatasetValidator(records=records, images_dir=images_dir)
    validator.validate() # Enriches records with valid_objects

    splitter = StratifiedDatasetSplitter(records=records, seed=seed)
    splits = splitter.create_splits(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    splitter.save_splits(splits, out_path)

    print("\n" + "="*60)
    print("DATASET SPLIT CREATION COMPLETE")
    print("="*60)
    print(f"Random Seed : {seed}")
    print(f"Train Set   : {len(splits['train'])} images")
    print(f"Val Set     : {len(splits['val'])} images")
    print(f"Test Set    : {len(splits['test'])} images")
    print(f"Saved to    : {out_path.resolve()}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create deterministic train/val/test splits.")
    parser.add_argument("--data_dir", type=str, default="data/raw/NEU-DET", help="Path to raw dataset directory")
    parser.add_argument("--output_dir", type=str, default="data/splits", help="Directory to save split text files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    main(args.data_dir, args.output_dir, args.seed)
