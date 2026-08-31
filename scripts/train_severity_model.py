import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.severity.index_calculator import TransparentSeverityIndexCalculator
from src.severity.ml_models import SeverityMLModelTrainer

logger = get_logger("train_severity")

FEATURE_COLS = [
    "defect_area_px", "bbox_area_px", "area_ratio", "bbox_fill_ratio",
    "perimeter", "length_px", "width_px", "aspect_ratio", "circularity",
    "compactness", "eccentricity", "orientation", "solidity", "extent",
    "equivalent_diameter", "connected_components_count", "mean_intensity",
    "std_intensity", "skewness", "kurtosis", "min_intensity", "max_intensity",
    "intensity_range", "local_contrast", "glcm_contrast", "glcm_correlation",
    "glcm_energy", "glcm_homogeneity", "lbp_mean", "lbp_std"
]

def main(features_csv: str | Path, splits_dir: str | Path, model_type: str, output_model: str | Path):
    df = pd.read_csv(features_csv)
    splits_path = Path(splits_dir)

    train_files = set(Path(f).name for f in open(splits_path / "train.txt").read().splitlines() if f.strip())
    val_files = set(Path(f).name for f in open(splits_path / "val.txt").read().splitlines() if f.strip())

    # Calculate severity ground truth index and categories
    calc = TransparentSeverityIndexCalculator()
    scores, categories, cat_ids = [], [], []

    for _, row in df.iterrows():
        s = calc.compute_severity_score(row)
        cat_name, cat_id = calc.get_severity_category(s)
        scores.append(s)
        categories.append(cat_name)
        cat_ids.append(cat_id)

    df["severity_score"] = scores
    df["severity_category"] = categories
    df["severity_id"] = cat_ids

    # Train / Val splits
    train_df = df[df["image_filename"].isin(train_files)].copy()
    val_df = df[df["image_filename"].isin(val_files)].copy()

    logger.info(f"Loaded feature dataset ({len(df)} samples). Train: {len(train_df)}, Val: {len(val_df)}")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["severity_id"].values

    trainer = SeverityMLModelTrainer(model_type=model_type, random_state=42)
    trainer.train(X_train, y_train)
    trainer.save(output_model)

    logger.info(f"Trained {model_type} severity model and saved to {output_model}")

    print("\n" + "="*60)
    print(f"SEVERITY MODEL TRAINING COMPLETE ({model_type.upper()})")
    print("="*60)
    print(f"Train samples : {len(X_train)}")
    print(f"Model Saved   : {Path(output_model).resolve()}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train feature-based defect severity ML model.")
    parser.add_argument("--features_csv", type=str, default="data/processed/defect_features.csv", help="Path to extracted features CSV")
    parser.add_argument("--splits_dir", type=str, default="data/splits", help="Path to splits directory")
    parser.add_argument("--model_type", type=str, default="random_forest", choices=["random_forest", "xgboost", "ridge"], help="ML model architecture")
    parser.add_argument("--output_model", type=str, default="models/severity_rf.pkl", help="Path to save trained model pickle")
    args = parser.parse_args()

    main(args.features_csv, args.splits_dir, args.model_type, args.output_model)
