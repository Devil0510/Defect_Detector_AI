"""
Robustness Testing & Expected Calibration Error (ECE) Analysis.

Evaluates pipeline degradation under synthetic corruptions:
  - Gaussian blur (σ = {1, 3, 5})
  - Gaussian noise (σ = {10, 25, 50})
  - Brightness shifts (δ = {-40, -20, +20, +40})

Computes Expected Calibration Error (ECE) for ordinal severity classification
and plots degradation curves showing how MAE and ECE change with corruption severity.
"""

import os
import sys
import json
import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.severity.neural_severity import EfficientNetV2SeverityModel
from src.severity.uncertainty import MonteCarloUncertaintyEstimator
from src.severity.index_calculator import TransparentSeverityIndexCalculator

logger = get_logger("robustness_test")

# ─────────────────────────────────────────────────────────────────────
# Synthetic Corruption Functions
# ─────────────────────────────────────────────────────────────────────

def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur with kernel size derived from sigma."""
    ksize = int(2 * round(3 * sigma) + 1)
    ksize = max(ksize, 3)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """Add zero-mean Gaussian noise with given standard deviation."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_brightness_shift(image: np.ndarray, delta: int) -> np.ndarray:
    """Shift pixel intensities by a constant delta (can be negative for darkening)."""
    shifted = image.astype(np.int16) + delta
    return np.clip(shifted, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
# Expected Calibration Error (ECE)
# ─────────────────────────────────────────────────────────────────────

def compute_ece(confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = Σ_{b=1}^{B} (n_b / N) × |acc(b) - conf(b)|
    
    Where:
    - B = number of bins
    - n_b = number of samples in bin b
    - acc(b) = accuracy of predictions in bin b
    - conf(b) = mean confidence of predictions in bin b
    """
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


# ─────────────────────────────────────────────────────────────────────
# Core Evaluation Under Corruption
# ─────────────────────────────────────────────────────────────────────

def evaluate_under_corruption(
    corruption_fn,
    corruption_param,
    corruption_name: str,
    model: EfficientNetV2SeverityModel,
    uncertainty_estimator: MonteCarloUncertaintyEstimator,
    test_df: pd.DataFrame,
    images_dir: Path,
    calc: TransparentSeverityIndexCalculator,
    device: torch.device,
    normalize: transforms.Normalize
) -> dict:
    """Run full severity evaluation on corrupted versions of test images."""
    
    y_true_scores, y_true_cats = [], []
    y_pred_means, y_pred_stds, y_pred_cats = [], [], []
    all_confidences = []
    
    img_cache = {}
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        img_name = row["image_filename"]
        
        # Load and corrupt the image
        if img_name not in img_cache:
            p = images_dir / img_name
            im = cv2.imread(str(p))
            if im is not None:
                corrupted = corruption_fn(im, corruption_param)
                img_cache[img_name] = cv2.cvtColor(corrupted, cv2.COLOR_BGR2RGB)
            else:
                img_cache[img_name] = np.zeros((200, 200, 3), dtype=np.uint8)
        
        image = img_cache[img_name]
        h, w = image.shape[:2]
        
        # Extract crop with 10px padding
        pad = 10
        x1 = max(0, int(round(row["bbox_xmin"])) - pad)
        y1 = max(0, int(round(row["bbox_ymin"])) - pad)
        x2 = min(w, int(round(row["bbox_xmax"])) + pad)
        y2 = min(h, int(round(row["bbox_ymax"])) + pad)
        
        crop = image[y1:y2, x1:x2] if (x2 - x1) >= 2 and (y2 - y1) >= 2 else image
        crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensor_crop = normalize(transforms.ToTensor()(crop_resized)).unsqueeze(0).to(device)
        
        # MC Dropout uncertainty estimation
        unc_res = uncertainty_estimator.estimate_uncertainty(tensor_crop)
        
        # Ground truth
        s_true = calc.compute_severity_score(row)
        _, cid_true = calc.get_severity_category(s_true)
        
        y_true_scores.append(s_true)
        y_true_cats.append(cid_true)
        y_pred_means.append(unc_res["mean_severity_score"])
        y_pred_stds.append(unc_res["std_severity_score"])
        y_pred_cats.append(unc_res["predicted_class_id"])
        
        # Confidence = max class probability from MC averaged softmax
        max_prob = max(unc_res["class_probabilities"])
        all_confidences.append(max_prob)
    
    y_true_s = np.array(y_true_scores)
    y_pred_s = np.array(y_pred_means)
    y_true_c = np.array(y_true_cats)
    y_pred_c = np.array(y_pred_cats)
    confidences = np.array(all_confidences)
    
    mae = float(mean_absolute_error(y_true_s, y_pred_s))
    acc = float(accuracy_score(y_true_c, y_pred_c))
    macro_f1 = float(f1_score(y_true_c, y_pred_c, average="macro", zero_division=0))
    sr, _ = spearmanr(y_true_s, y_pred_s)
    mean_std = float(np.mean(y_pred_stds))
    ece = compute_ece(confidences, y_pred_c, y_true_c, n_bins=10)
    
    return {
        "corruption_type": corruption_name,
        "corruption_param": corruption_param,
        "mae": mae,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "spearman_rho": float(sr),
        "mean_uncertainty_std": mean_std,
        "ece": ece
    }


# ─────────────────────────────────────────────────────────────────────
# Degradation Curve Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_degradation_curves(results: list, output_dir: Path):
    """Generate degradation curve plots for each corruption type."""
    
    corruption_groups = {}
    for r in results:
        ctype = r["corruption_type"]
        if ctype not in corruption_groups:
            corruption_groups[ctype] = []
        corruption_groups[ctype].append(r)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Pipeline Robustness Under Synthetic Corruptions", fontsize=14, fontweight="bold")
    
    colors = {"Gaussian Blur": "#2196F3", "Gaussian Noise": "#F44336", "Brightness Shift": "#4CAF50"}
    
    # Plot 1: MAE Degradation
    ax1 = axes[0]
    for ctype, group in corruption_groups.items():
        params = [g["corruption_param"] for g in group]
        maes = [g["mae"] for g in group]
        ax1.plot(range(len(params)), maes, "o-", color=colors.get(ctype, "gray"), label=ctype, linewidth=2, markersize=8)
        ax1.set_xticks(range(len(params)))
        ax1.set_xticklabels([str(p) for p in params], fontsize=9)
    ax1.set_xlabel("Corruption Parameter", fontsize=11)
    ax1.set_ylabel("Severity Score MAE (points)", fontsize=11)
    ax1.set_title("MAE Degradation", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ECE Degradation
    ax2 = axes[1]
    for ctype, group in corruption_groups.items():
        params = [g["corruption_param"] for g in group]
        eces = [g["ece"] for g in group]
        ax2.plot(range(len(params)), eces, "s-", color=colors.get(ctype, "gray"), label=ctype, linewidth=2, markersize=8)
        ax2.set_xticks(range(len(params)))
        ax2.set_xticklabels([str(p) for p in params], fontsize=9)
    ax2.set_xlabel("Corruption Parameter", fontsize=11)
    ax2.set_ylabel("Expected Calibration Error (ECE)", fontsize=11)
    ax2.set_title("Calibration Degradation", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty Increase
    ax3 = axes[2]
    for ctype, group in corruption_groups.items():
        params = [g["corruption_param"] for g in group]
        stds = [g["mean_uncertainty_std"] for g in group]
        ax3.plot(range(len(params)), stds, "^-", color=colors.get(ctype, "gray"), label=ctype, linewidth=2, markersize=8)
        ax3.set_xticks(range(len(params)))
        ax3.set_xticklabels([str(p) for p in params], fontsize=9)
    ax3.set_xlabel("Corruption Parameter", fontsize=11)
    ax3.set_ylabel("Mean Epistemic σ (points)", fontsize=11)
    ax3.set_title("Uncertainty Response", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "robustness_degradation_curves.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved degradation curves to: {plot_path}")


def plot_reliability_diagram(results: list, output_dir: Path):
    """Plot reliability diagram (calibration curve) for the clean baseline."""
    # Use clean baseline result (no corruption)
    clean = [r for r in results if r["corruption_type"] == "Clean (No Corruption)"]
    if not clean:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)
    
    ece_val = clean[0]["ece"]
    ax.text(0.05, 0.90, f"ECE = {ece_val:.4f}", transform=ax.transAxes, fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="orange"))
    
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction of Correct Predictions", fontsize=12)
    ax.set_title("Reliability Diagram (Clean Baseline)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "reliability_diagram_clean.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved reliability diagram to: {plot_path}")


# ─────────────────────────────────────────────────────────────────────
# Main Entry
# ─────────────────────────────────────────────────────────────────────

def run_robustness_test(
    model_path: str,
    features_csv: str,
    splits_dir: str,
    images_dir: str,
    output_dir: str
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test split
    df = pd.read_csv(features_csv)
    test_files = set(Path(f).name for f in open(Path(splits_dir) / "test.txt").read().splitlines() if f.strip())
    
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
    logger.info(f"Robustness test on {len(test_df)} test samples")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetV2SeverityModel(pretrained=False, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    unc_est = MonteCarloUncertaintyEstimator(model, n_samples=25, uncertainty_threshold=8.0)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    images_path = Path(images_dir)
    
    # Define corruption schedule
    corruptions = [
        ("Clean (No Corruption)", lambda img, _: img.copy(), [0]),
        ("Gaussian Blur", apply_gaussian_blur, [1, 3, 5]),
        ("Gaussian Noise", apply_gaussian_noise, [10, 25, 50]),
        ("Brightness Shift", apply_brightness_shift, [-40, -20, 20, 40]),
    ]
    
    all_results = []
    
    for corr_name, corr_fn, params in corruptions:
        for param in params:
            label = f"{corr_name} (param={param})" if corr_name != "Clean (No Corruption)" else corr_name
            logger.info(f"Evaluating: {label}...")
            
            result = evaluate_under_corruption(
                corruption_fn=corr_fn,
                corruption_param=param,
                corruption_name=corr_name,
                model=model,
                uncertainty_estimator=unc_est,
                test_df=test_df,
                images_dir=images_path,
                calc=calc,
                device=device,
                normalize=normalize
            )
            all_results.append(result)
            
            print(f"  {label}: MAE={result['mae']:.2f}, ECE={result['ece']:.4f}, "
                  f"Acc={result['accuracy']*100:.1f}%, σ={result['mean_uncertainty_std']:.2f}")
    
    # Save JSON report
    report_path = out_dir / "robustness_results.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Saved robustness report: {report_path}")
    
    # Generate plots
    plot_degradation_curves(all_results, out_dir)
    plot_reliability_diagram(all_results, out_dir)
    
    # Print summary table
    print("\n" + "=" * 90)
    print("ROBUSTNESS & CALIBRATION TEST RESULTS")
    print("=" * 90)
    print(f"{'Corruption':<30} {'Param':>6} {'MAE':>8} {'ECE':>8} {'Acc%':>8} {'F1':>8} {'σ':>8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['corruption_type']:<30} {r['corruption_param']:>6} {r['mae']:>8.2f} {r['ece']:>8.4f} "
              f"{r['accuracy']*100:>7.1f}% {r['macro_f1']:>8.4f} {r['mean_uncertainty_std']:>8.2f}")
    print("=" * 90)
    print(f"Results saved to: {out_dir.resolve()}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness testing and ECE calibration analysis.")
    parser.add_argument("--model_path", type=str, default="models/severity_efficientnet_v2.pt")
    parser.add_argument("--features_csv", type=str, default="data/processed/defect_features.csv")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--images_dir", type=str, default="data/raw/NEU-DET/IMAGES")
    parser.add_argument("--output_dir", type=str, default="reports/robustness")
    args = parser.parse_args()
    
    run_robustness_test(args.model_path, args.features_csv, args.splits_dir, args.images_dir, args.output_dir)
