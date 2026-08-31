import os
import sys
import json
import cv2
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, classification_report
from tqdm import tqdm

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.severity.neural_severity import EfficientNetV2SeverityModel
from src.severity.uncertainty import MonteCarloUncertaintyEstimator
from src.severity.index_calculator import TransparentSeverityIndexCalculator

logger = get_logger("evaluate_neural_severity")

def evaluate_neural_model(model_path: str | Path, features_csv: str | Path, splits_dir: str | Path, images_dir: str | Path, output_report: str | Path):
    df = pd.read_csv(features_csv)
    splits_path = Path(splits_dir)
    images_path = Path(images_dir)

    test_files = set(Path(f).name for f in open(splits_path / "test.txt").read().splitlines() if f.strip())

    calc = TransparentSeverityIndexCalculator()
    scores, cat_ids = [], []
    for _, row in df.iterrows():
        s = calc.compute_severity_score(row)
        _, cid = calc.get_severity_category(s)
        scores.append(s)
        cat_ids.append(cid)

    df["severity_score"] = scores
    df["severity_id"] = cat_ids

    test_df = df[df["image_filename"].isin(test_files)].reset_index(drop=True)
    logger.info(f"Evaluating EfficientNet-V2 on held-out test split ({len(test_df)} samples)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetV2SeverityModel(pretrained=False, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))

    uncertainty_estimator = MonteCarloUncertaintyEstimator(model, n_samples=25, uncertainty_threshold=8.0)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    y_true_scores, y_true_cats = [], []
    y_pred_means, y_pred_stds, y_pred_cats = [], [], []
    ci_lower_list, ci_upper_list = [], []
    triage_flags = []

    img_cache = {}

    for i in tqdm(range(len(test_df)), desc="Evaluating MC Dropout"):
        row = test_df.iloc[i]
        img_name = row["image_filename"]
        if img_name not in img_cache:
            p = images_path / img_name
            im = cv2.imread(str(p))
            img_cache[img_name] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else np.zeros((200, 200, 3), dtype=np.uint8)

        image = img_cache[img_name]
        h, w = image.shape[:2]
        x1 = max(0, int(round(row["bbox_xmin"])))
        y1 = max(0, int(round(row["bbox_ymin"])))
        x2 = min(w, int(round(row["bbox_xmax"])))
        y2 = min(h, int(round(row["bbox_ymax"])))

        crop = image[y1:y2, x1:x2] if (x2 - x1) >= 2 and (y2 - y1) >= 2 else image
        crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensor_crop = normalize(transforms.ToTensor()(crop_resized)).unsqueeze(0).to(device)

        unc_res = uncertainty_estimator.estimate_uncertainty(tensor_crop)

        y_true_scores.append(float(row["severity_score"]))
        y_true_cats.append(int(row["severity_id"]))

        y_pred_means.append(unc_res["mean_severity_score"])
        y_pred_stds.append(unc_res["std_severity_score"])
        y_pred_cats.append(unc_res["predicted_class_id"])
        ci_lower_list.append(unc_res["ci_95_lower"])
        ci_upper_list.append(unc_res["ci_95_upper"])
        triage_flags.append(unc_res["requires_human_triage"])

    y_true_s = np.array(y_true_scores)
    y_pred_s = np.array(y_pred_means)
    y_std_s = np.array(y_pred_stds)

    score_mae = float(mean_absolute_error(y_true_s, y_pred_s))
    pr, _ = pearsonr(y_true_s, y_pred_s)
    sr, _ = spearmanr(y_true_s, y_pred_s)

    cat_acc = float(accuracy_score(y_true_cats, y_pred_cats))
    macro_f1 = float(f1_score(y_true_cats, y_pred_cats, average="macro", zero_division=0))

    # Empirical 95% Credible Interval Coverage
    in_ci = (y_true_s >= np.array(ci_lower_list)) & (y_true_s <= np.array(ci_upper_list))
    ci_coverage_pct = float(np.mean(in_ci) * 100.0)

    mean_uncertainty_std = float(np.mean(y_std_s))
    triage_rate_pct = float(np.mean(triage_flags) * 100.0)

    metrics = {
        "model_architecture": "EfficientNet-V2-S Dual-Head",
        "test_samples_count": len(test_df),
        "continuous_score_mae_points": score_mae,
        "score_pearson_r": float(pr),
        "score_spearman_rho": float(sr),
        "ordinal_classification_accuracy": cat_acc,
        "ordinal_macro_f1": macro_f1,
        "mean_epistemic_uncertainty_std": mean_uncertainty_std,
        "ci_95_empirical_coverage_pct": ci_coverage_pct,
        "human_triage_flag_rate_pct": triage_rate_pct
    }

    out_p = Path(output_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "="*70)
    print("EFFICIENTNET-V2 SEVERITY & UNCERTAINTY TEST EVALUATION")
    print("="*70)
    print(f"Continuous Score MAE       : {score_mae:.2f} points (out of 100)")
    print(f"Spearman Rank Correlation  : {sr:.4f}")
    print(f"Pearson Correlation (r)    : {pr:.4f}")
    print(f"Ordinal Category Accuracy  : {cat_acc*100:.2f}%")
    print(f"Macro F1 Score             : {macro_f1:.4f}")
    print(f"Mean Predictive Std (σ)    : ±{mean_uncertainty_std:.2f} points")
    print(f"95% CI Empirical Coverage  : {ci_coverage_pct:.1f}%")
    print(f"Human Triage Flag Rate     : {triage_rate_pct:.1f}% of samples")
    print(f"Saved Report               : {out_p.resolve()}")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-V2 neural severity model and uncertainty.")
    parser.add_argument("--model_path", type=str, default="models/severity_efficientnet_v2.pt")
    parser.add_argument("--features_csv", type=str, default="data/processed/defect_features.csv")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--images_dir", type=str, default="data/raw/NEU-DET/IMAGES")
    parser.add_argument("--output_report", type=str, default="reports/neural_severity_metrics.json")
    args = parser.parse_args()

    evaluate_neural_model(args.model_path, args.features_csv, args.splits_dir, args.images_dir, args.output_report)
