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
from src.severity.uncertainty import (
    MonteCarloUncertaintyEstimator,
    ConformalPredictionCalibrator,
    TemperatureScaler
)
from src.severity.index_calculator import TransparentSeverityIndexCalculator

logger = get_logger("evaluate_neural_severity")

def compute_ece(confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)
    if total == 0:
        return 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        n_in_bin = np.sum(in_bin)
        if n_in_bin == 0:
            continue
        bin_acc = np.mean(predictions[in_bin] == labels[in_bin])
        bin_conf = np.mean(confidences[in_bin])
        ece += (n_in_bin / total) * abs(bin_acc - bin_conf)
    return float(ece)

def evaluate_and_calibrate(
    model_path: str | Path,
    features_csv: str | Path,
    splits_dir: str | Path,
    images_dir: str | Path,
    output_report: str | Path,
    conformal_save_path: str | Path = "models/conformal_calibrator.json",
    temperature_save_path: str | Path = "models/temperature_scaler.json"
):
    df = pd.read_csv(features_csv)
    splits_path = Path(splits_dir)
    images_path = Path(images_dir)

    val_files = set(Path(f).name for f in open(splits_path / "val.txt").read().splitlines() if f.strip())
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

    val_df = df[df["image_filename"].isin(val_files)].reset_index(drop=True)
    test_df = df[df["image_filename"].isin(test_files)].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetV2SeverityModel(pretrained=False, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_cache = {}

    def get_crop_tensor(row):
        img_name = row["image_filename"]
        if img_name not in img_cache:
            p = images_path / img_name
            im = cv2.imread(str(p))
            img_cache[img_name] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else np.zeros((200, 200, 3), dtype=np.uint8)
        image = img_cache[img_name]
        h, w = image.shape[:2]
        pad = 10
        x1 = max(0, int(round(row["bbox_xmin"])) - pad)
        y1 = max(0, int(round(row["bbox_ymin"])) - pad)
        x2 = min(w, int(round(row["bbox_xmax"])) + pad)
        y2 = min(h, int(round(row["bbox_ymax"])) + pad)
        crop = image[y1:y2, x1:x2] if (x2 - x1) >= 2 and (y2 - y1) >= 2 else image
        crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        return normalize(transforms.ToTensor()(crop_resized)).unsqueeze(0).to(device)

    # ─────────────────────────────────────────────────────────────────
    # Phase 1: Calibrate Conformal Predictor & Temperature on Val Set
    # ─────────────────────────────────────────────────────────────────
    logger.info(f"Phase 1: Calibrating Conformal Predictor & Temperature Scaler on Validation Set ({len(val_df)} samples)...")
    val_raw_unc = MonteCarloUncertaintyEstimator(model, n_samples=25)
    
    val_true_scores = val_df["severity_score"].to_numpy()
    val_true_cats = val_df["severity_id"].to_numpy()
    val_pred_means, val_pred_stds = [], []
    val_logits_list = []

    batch_size = 32
    for start_idx in tqdm(range(0, len(val_df), batch_size), desc="Validation Batches"):
        end_idx = min(start_idx + batch_size, len(val_df))
        batch_crops = [get_crop_tensor(val_df.iloc[k]) for k in range(start_idx, end_idx)]
        batch_tensor = torch.cat(batch_crops, dim=0)

        res = val_raw_unc.estimate_uncertainty_batch(batch_tensor)
        val_pred_means.extend(res["mean_severity_scores"].tolist())
        val_pred_stds.extend(res["std_severity_scores"].tolist())
        val_logits_list.extend(res["mean_logits"].tolist())

    val_true_s = np.array(val_true_scores)
    val_pred_m = np.array(val_pred_means)
    val_pred_s = np.array(val_pred_stds)
    val_logits_arr = np.array(val_logits_list)
    val_true_c = np.array(val_true_cats)

    # Fit Conformal Calibrator
    conformal_calibrator = ConformalPredictionCalibrator(alpha=0.05)
    q_hat = conformal_calibrator.calibrate(val_true_s, val_pred_m, val_pred_s)
    conformal_calibrator.save(conformal_save_path)
    logger.info(f"--> Conformal 95% Quantile Multiplier q_hat calibrated on Val Set: {q_hat:.4f} (Saved to {conformal_save_path})")

    # Fit Temperature Scaler
    temp_scaler = TemperatureScaler()
    opt_temp = temp_scaler.fit(val_logits_arr, val_true_c)
    temp_scaler.save(temperature_save_path)
    logger.info(f"--> Optimal Temperature T fitted on Val Set: {opt_temp:.4f} (Saved to {temperature_save_path})")

    # ─────────────────────────────────────────────────────────────────
    # Phase 2: Comprehensive Evaluation on Held-Out Test Split
    # ─────────────────────────────────────────────────────────────────
    logger.info(f"Phase 2: Evaluating Calibrated Pipeline on Test Split ({len(test_df)} samples)...")
    calibrated_estimator = MonteCarloUncertaintyEstimator(
        model=model,
        n_samples=25,
        uncertainty_threshold=8.0,
        conformal_calibrator=conformal_calibrator,
        temperature_scaler=temp_scaler
    )

    test_true_scores = test_df["severity_score"].to_numpy()
    test_true_cats = test_df["severity_id"].to_numpy()
    test_pred_means, test_pred_stds, test_pred_cats = [], [], []
    raw_ci_lower, raw_ci_upper = [], []
    conf_ci_lower, conf_ci_upper = [], []
    test_raw_probs, test_cal_probs = [], []
    triage_flags = []

    for start_idx in tqdm(range(0, len(test_df), batch_size), desc="Test Batches"):
        end_idx = min(start_idx + batch_size, len(test_df))
        batch_crops = [get_crop_tensor(test_df.iloc[k]) for k in range(start_idx, end_idx)]
        batch_tensor = torch.cat(batch_crops, dim=0)

        res = calibrated_estimator.estimate_uncertainty_batch(batch_tensor)

        test_pred_means.extend(res["mean_severity_scores"].tolist())
        test_pred_stds.extend(res["std_severity_scores"].tolist())
        test_pred_cats.extend(res["predicted_class_ids"].tolist())

        raw_ci_lower.extend(res["raw_mc_ci_95_lowers"].tolist())
        raw_ci_upper.extend(res["raw_mc_ci_95_uppers"].tolist())
        conf_ci_lower.extend(res["ci_95_lowers"].tolist())
        conf_ci_upper.extend(res["ci_95_uppers"].tolist())

        test_cal_probs.extend(res["class_probabilities"].tolist())
        triage_flags.extend(res["requires_human_triage"].tolist())

        with torch.no_grad():
            _, logits = model(batch_tensor, enable_mc_dropout=False)
            uncal_p = torch.softmax(logits, dim=-1).cpu().numpy()
            test_raw_probs.extend(uncal_p.tolist())


    test_true_s = np.array(test_true_scores)
    test_pred_m = np.array(test_pred_means)
    test_pred_s = np.array(test_pred_stds)
    test_true_c = np.array(test_true_cats)
    test_pred_c = np.array(test_pred_cats)

    score_mae = float(mean_absolute_error(test_true_s, test_pred_m))
    pr, _ = pearsonr(test_true_s, test_pred_m)
    sr, _ = spearmanr(test_true_s, test_pred_m)

    cat_acc = float(accuracy_score(test_true_c, test_pred_c))
    macro_f1 = float(f1_score(test_true_c, test_pred_c, average="macro", zero_division=0))

    # Interval Coverage
    in_raw_ci = (test_true_s >= np.array(raw_ci_lower)) & (test_true_s <= np.array(raw_ci_upper))
    raw_ci_coverage_pct = float(np.mean(in_raw_ci) * 100.0)

    in_conf_ci = (test_true_s >= np.array(conf_ci_lower)) & (test_true_s <= np.array(conf_ci_upper))
    conformal_ci_coverage_pct = float(np.mean(in_conf_ci) * 100.0)

    # Expected Calibration Error (ECE)
    test_raw_probs_arr = np.array(test_raw_probs)
    test_cal_probs_arr = np.array(test_cal_probs)

    uncal_conf = np.max(test_raw_probs_arr, axis=-1)
    uncal_preds = np.argmax(test_raw_probs_arr, axis=-1)
    ece_uncalibrated = compute_ece(uncal_conf, uncal_preds, test_true_c)

    cal_conf = np.max(test_cal_probs_arr, axis=-1)
    cal_preds = np.argmax(test_cal_probs_arr, axis=-1)
    ece_calibrated = compute_ece(cal_conf, cal_preds, test_true_c)

    mean_uncertainty_std = float(np.mean(test_pred_s))
    triage_rate_pct = float(np.mean(triage_flags) * 100.0)

    metrics = {
        "model_architecture": "EfficientNet-V2-S Dual-Head (End-to-End Fine-Tuned)",
        "test_samples_count": len(test_df),
        "continuous_score_mae_points": score_mae,
        "score_pearson_r": float(pr),
        "score_spearman_rho": float(sr),
        "ordinal_classification_accuracy": cat_acc,
        "ordinal_macro_f1": macro_f1,
        "mean_epistemic_uncertainty_std": mean_uncertainty_std,
        "raw_mc_ci_95_empirical_coverage_pct": raw_ci_coverage_pct,
        "conformal_ci_95_empirical_coverage_pct": conformal_ci_coverage_pct,
        "conformal_q95_multiplier": float(q_hat),
        "ece_uncalibrated": ece_uncalibrated,
        "ece_temperature_scaled": ece_calibrated,
        "temperature": opt_temp,
        "human_triage_flag_rate_pct": triage_rate_pct
    }

    out_p = Path(output_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "="*75)
    print("EFFICIENTNET-V2 SEVERITY, CONFORMAL PREDICTION & CALIBRATION REPORT")
    print("="*75)
    print(f"Continuous Score MAE         : {score_mae:.2f} points (out of 100)")
    print(f"Spearman Rank Correlation (ρ): {sr:.4f}")
    print(f"Pearson Correlation (r)      : {pr:.4f}")
    print(f"Ordinal Category Accuracy    : {cat_acc*100:.2f}%")
    print(f"Macro F1 Score               : {macro_f1:.4f}")
    print(f"Mean Predictive Std (σ)      : ±{mean_uncertainty_std:.2f} points")
    print(f"Uncalibrated 95% MC Coverage : {raw_ci_coverage_pct:.1f}%")
    print(f"Conformal 95% Coverage       : {conformal_ci_coverage_pct:.1f}% (Guaranteed >= 95%)")
    print(f"Conformal Multiplier (q̂_0.95): {q_hat:.4f}")
    print(f"ECE (Uncalibrated vs Scaled) : {ece_uncalibrated:.4f} -> {ece_calibrated:.4f}")
    print(f"Optimal Temperature (T)      : {opt_temp:.4f}")
    print(f"Human Triage Flag Rate       : {triage_rate_pct:.1f}% of samples")
    print(f"Saved Report                 : {out_p.resolve()}")
    print("="*75 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-V2 neural severity model with conformal prediction and temperature scaling.")
    parser.add_argument("--model_path", type=str, default="models/severity_efficientnet_v2.pt")
    parser.add_argument("--features_csv", type=str, default="data/processed/defect_features.csv")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--images_dir", type=str, default="data/raw/NEU-DET/IMAGES")
    parser.add_argument("--output_report", type=str, default="reports/neural_severity_metrics.json")
    parser.add_argument("--conformal_save_path", type=str, default="models/conformal_calibrator.json")
    parser.add_argument("--temperature_save_path", type=str, default="models/temperature_scaler.json")
    args = parser.parse_args()

    evaluate_and_calibrate(
        args.model_path,
        args.features_csv,
        args.splits_dir,
        args.images_dir,
        args.output_report,
        args.conformal_save_path,
        args.temperature_save_path
    )

