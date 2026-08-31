import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.severity.index_calculator import TransparentSeverityIndexCalculator
from src.severity.ml_models import SeverityMLModelTrainer
from src.severity.ordinal_evaluator import OrdinalSeverityEvaluator
from scripts.train_severity_model import FEATURE_COLS

logger = get_logger("evaluate_severity")

def main(features_csv: str | Path, splits_dir: str | Path, model_file: str | Path, output_report: str | Path):
    df = pd.read_csv(features_csv)
    splits_path = Path(splits_dir)

    test_files = set(Path(f).name for f in open(splits_path / "test.txt").read().splitlines() if f.strip())

    # Calculate severity ground truth index and categories
    calc = TransparentSeverityIndexCalculator()
    cat_ids = []

    for _, row in df.iterrows():
        s = calc.compute_severity_score(row)
        _, cat_id = calc.get_severity_category(s)
        cat_ids.append(cat_id)

    df["severity_id"] = cat_ids
    test_df = df[df["image_filename"].isin(test_files)].copy()

    logger.info(f"Loaded held-out test split ({len(test_df)} samples) for severity evaluation.")

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["severity_id"].values

    trainer = SeverityMLModelTrainer()
    trainer.load(model_file)

    preds = trainer.predict(X_test)
    metrics = OrdinalSeverityEvaluator.evaluate(y_test, preds)

    out_path = Path(output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "="*60)
    print("SEVERITY MODEL TEST EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy                  : {metrics['accuracy']:.4f}")
    print(f"Macro F1                  : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1               : {metrics['weighted_f1']:.4f}")
    print(f"Ordinal MAE               : {metrics['ordinal_mae']:.4f}")
    print(f"Quadratic Weighted Kappa  : {metrics['quadratic_weighted_kappa']:.4f}")
    print(f"Spearman Rho              : {metrics['spearman_rho']:.4f}")
    print(f"Kendall Tau               : {metrics['kendall_tau']:.4f}")
    print(f"Saved Report              : {out_path.resolve()}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate severity estimation model on test split.")
    parser.add_argument("--features_csv", type=str, default="data/processed/defect_features.csv", help="Path to extracted features CSV")
    parser.add_argument("--splits_dir", type=str, default="data/splits", help="Path to splits directory")
    parser.add_argument("--model_file", type=str, default="models/severity_rf.pkl", help="Path to trained model pickle")
    parser.add_argument("--output_report", type=str, default="reports/severity_metrics.json", help="Path to save evaluation report JSON")
    args = parser.parse_args()

    main(args.features_csv, args.splits_dir, args.model_file, args.output_report)
