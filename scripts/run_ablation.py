import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.severity.index_calculator import TransparentSeverityIndexCalculator
from src.severity.ml_models import SeverityMLModelTrainer
from src.severity.ordinal_evaluator import OrdinalSeverityEvaluator

logger = get_logger("run_ablation")

GEOM_COLS = ["defect_area_px", "bbox_area_px", "area_ratio", "bbox_fill_ratio", "perimeter", "length_px", "width_px", "aspect_ratio", "circularity", "compactness", "eccentricity", "orientation", "solidity", "extent", "equivalent_diameter", "connected_components_count"]
INTEN_COLS = ["mean_intensity", "std_intensity", "skewness", "kurtosis", "min_intensity", "max_intensity", "intensity_range", "local_contrast"]
TEX_COLS = ["glcm_contrast", "glcm_correlation", "glcm_energy", "glcm_homogeneity", "lbp_mean", "lbp_std"]

def main(features_csv: str | Path, splits_dir: str | Path, output_dir: str | Path):
    df = pd.read_csv(features_csv)
    splits_path = Path(splits_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_files = set(Path(f).name for f in open(splits_path / "train.txt").read().splitlines() if f.strip())
    test_files = set(Path(f).name for f in open(splits_path / "test.txt").read().splitlines() if f.strip())

    calc = TransparentSeverityIndexCalculator()
    cat_ids = [calc.get_severity_category(calc.compute_severity_score(row))[1] for _, row in df.iterrows()]
    df["severity_id"] = cat_ids

    train_df = df[df["image_filename"].isin(train_files)].copy()
    test_df = df[df["image_filename"].isin(test_files)].copy()

    experiments = {
        "Exp_1_BBox_Only": ["bbox_area_px", "length_px", "width_px"],
        "Exp_2_Geometry": GEOM_COLS,
        "Exp_3_Geom_Intensity": GEOM_COLS + INTEN_COLS,
        "Exp_4_Full_Features": GEOM_COLS + INTEN_COLS + TEX_COLS
    }

    results = {}

    for exp_name, feat_list in experiments.items():
        X_tr = train_df[feat_list].values
        y_tr = train_df["severity_id"].values
        X_te = test_df[feat_list].values
        y_te = test_df["severity_id"].values

        trainer = SeverityMLModelTrainer(model_type="random_forest", random_state=42)
        trainer.train(X_tr, y_tr)
        preds = trainer.predict(X_te)

        m = OrdinalSeverityEvaluator.evaluate(y_te, preds)
        results[exp_name] = m
        logger.info(f"{exp_name} -> Acc: {m['accuracy']:.4f}, Macro F1: {m['macro_f1']:.4f}, MAE: {m['ordinal_mae']:.4f}")

    # Model comparison experiment (Exp 5)
    model_types = ["random_forest", "xgboost", "ridge"]
    full_feats = GEOM_COLS + INTEN_COLS + TEX_COLS
    X_tr, y_tr = train_df[full_feats].values, train_df["severity_id"].values
    X_te, y_te = test_df[full_feats].values, test_df["severity_id"].values

    for m_type in model_types:
        exp_key = f"Exp_5_Model_{m_type}"
        trainer = SeverityMLModelTrainer(model_type=m_type, random_state=42)
        trainer.train(X_tr, y_tr)
        preds = trainer.predict(X_te)
        m = OrdinalSeverityEvaluator.evaluate(y_te, preds)
        results[exp_key] = m

    report_path = out_path / "ablation_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved ablation results to {report_path}")

    # Plot Ablation Chart
    plot_df = pd.DataFrame([
        {"Experiment": k, "Macro F1": v["macro_f1"], "Accuracy": v["accuracy"], "Ordinal MAE": v["ordinal_mae"]}
        for k, v in results.items()
    ])

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="Experiment", y="Macro F1", palette="viridis")
    plt.title("Feature Ablation Study — Severity Estimation (Macro F1 Score)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt_path = out_path / "ablation_performance.png"
    plt.savefig(plt_path, dpi=300)
    plt.close()
    logger.info(f"Saved ablation chart to {plt_path}")

    print("\n" + "="*70)
    print("FEATURE ABLATION STUDY COMPLETE")
    print("="*70)
    for k, v in results.items():
        print(f"{k:<25} | Acc: {v['accuracy']:.4f} | F1: {v['macro_f1']:.4f} | MAE: {v['ordinal_mae']:.4f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quantitative feature group ablation study.")
    parser.add_argument("--features_csv", type=str, default="data/processed/defect_features.csv", help="Path to extracted features CSV")
    parser.add_argument("--splits_dir", type=str, default="data/splits", help="Path to splits directory")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory to save ablation reports and plots")
    args = parser.parse_args()

    main(args.features_csv, args.splits_dir, args.output_dir)
