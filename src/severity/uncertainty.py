import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from scipy.optimize import minimize

def evaluate_qc_triage(
    detector_confidence: float,
    mc_dropout_std: float,
    conformal_interval_width: float,
    min_detector_confidence: float = 0.50,
    max_mc_std: float = 8.0,
    max_conformal_width: float = 30.0,
    moderate_detector_confidence: float = 0.70,
    moderate_mc_std: float = 5.0,
    moderate_conformal_width: float = 20.0
) -> Tuple[str, bool, List[str]]:
    """
    Evaluates multi-factor Quality Control (QC) triage status based on multiple uncertainty signals:
    1. Object detector confidence
    2. Epistemic uncertainty (MC Dropout predictive standard deviation)
    3. Conformal prediction interval width (upper - lower bound)

    Returns:
        qc_status: "HIGH_CONFIDENCE", "MODERATE_CONFIDENCE", or "HUMAN_REVIEW_REQUIRED"
        requires_human_triage: bool (True if HUMAN_REVIEW_REQUIRED)
        triage_reasons: List of reasons triggering human inspection
    """
    reasons = []

    if detector_confidence < min_detector_confidence:
        reasons.append(f"Low detector confidence ({detector_confidence * 100:.1f}% < {min_detector_confidence * 100:.0f}%)")

    if mc_dropout_std > max_mc_std:
        reasons.append(f"High epistemic uncertainty (σ = {mc_dropout_std:.2f} > {max_mc_std:.1f} pts)")

    if conformal_interval_width > max_conformal_width:
        reasons.append(f"Wide conformal interval (width = {conformal_interval_width:.1f} > {max_conformal_width:.1f} pts)")

    if reasons:
        return "HUMAN_REVIEW_REQUIRED", True, reasons

    # Check for Moderate Confidence
    is_moderate = (
        detector_confidence < moderate_detector_confidence or
        mc_dropout_std > moderate_mc_std or
        conformal_interval_width > moderate_conformal_width
    )

    if is_moderate:
        return "MODERATE_CONFIDENCE", False, []

    return "HIGH_CONFIDENCE", False, []


class ConformalPredictionCalibrator:
    """
    Split Conformal Prediction Calibrator for continuous severity estimation.
    Guarantees finite-sample 1-alpha marginal coverage for severity prediction intervals.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.target_coverage = 1.0 - self.alpha
        self.q_hat: Optional[float] = None
        self.is_calibrated: bool = False
        self.use_normalized_residuals: bool = False

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred_mean: np.ndarray,
        y_pred_std: Optional[np.ndarray] = None
    ) -> float:
        """
        Computes calibrated non-conformity quantile q_hat on calibration split.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred_mean = np.asarray(y_pred_mean, dtype=np.float64)
        n = len(y_true)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        residuals = np.abs(y_true - y_pred_mean)

        if y_pred_std is not None:
            y_pred_std = np.asarray(y_pred_std, dtype=np.float64)
            stds = np.clip(y_pred_std, 1e-4, None)
            norm_residuals = residuals / stds
            # Compute finite-sample conformal quantile with ceil((n+1)*(1-alpha))/n
            level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
            self.q_hat = float(np.quantile(norm_residuals, level))
            self.use_normalized_residuals = True
        else:
            level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
            self.q_hat = float(np.quantile(residuals, level))
            self.use_normalized_residuals = False

        self.is_calibrated = True
        return self.q_hat

    def predict_interval(
        self,
        y_pred_mean: float | np.ndarray,
        y_pred_std: Optional[float | np.ndarray] = None,
        min_val: float = 0.0,
        max_val: float = 100.0
    ) -> Tuple[np.ndarray | float, np.ndarray | float]:
        """
        Constructs valid 1-alpha prediction interval [lower, upper].
        """
        if not self.is_calibrated or self.q_hat is None:
            std = y_pred_std if y_pred_std is not None else 5.0
            margin = 1.96 * np.asarray(std)
        else:
            if self.use_normalized_residuals and y_pred_std is not None:
                margin = self.q_hat * np.asarray(y_pred_std)
            else:
                margin = self.q_hat

        mean_arr = np.asarray(y_pred_mean)
        lower = np.clip(mean_arr - margin, min_val, max_val)
        upper = np.clip(mean_arr + margin, min_val, max_val)

        if np.ndim(y_pred_mean) == 0:
            return float(lower), float(upper)
        return lower, upper

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "alpha": self.alpha,
            "target_coverage": self.target_coverage,
            "q_hat": self.q_hat,
            "is_calibrated": self.is_calibrated,
            "use_normalized_residuals": self.use_normalized_residuals
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load(self, path: str | Path) -> None:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.alpha = float(data.get("alpha", 0.05))
        self.target_coverage = float(data.get("target_coverage", 0.95))
        self.q_hat = float(data["q_hat"]) if data.get("q_hat") is not None else None
        self.is_calibrated = bool(data.get("is_calibrated", True))
        self.use_normalized_residuals = bool(data.get("use_normalized_residuals", False))


class TemperatureScaler:
    """
    Post-hoc probability calibration via temperature scaling for ordinal classification logits.
    """

    def __init__(self):
        self.temperature: float = 1.0
        self.is_fitted: bool = False

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Fits temperature parameter T by minimizing cross-entropy NLL on validation logits.
        """
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        def nll_loss(t_val):
            t = max(t_val[0], 1e-3)
            scaled_logits = logits / t
            max_l = np.max(scaled_logits, axis=-1, keepdims=True)
            log_sum_exp = max_l + np.log(np.sum(np.exp(scaled_logits - max_l), axis=-1, keepdims=True))
            log_probs = scaled_logits - log_sum_exp
            nll = -np.mean(log_probs[np.arange(len(labels)), labels])
            return nll

        res = minimize(nll_loss, [1.5], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        self.is_fitted = True
        return self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Scales logits by temperature and applies softmax.
        """
        logits = np.asarray(logits, dtype=np.float64)
        t = self.temperature if self.is_fitted else 1.0
        scaled = logits / max(t, 1e-3)
        max_l = np.max(scaled, axis=-1, keepdims=True)
        exp_l = np.exp(scaled - max_l)
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        return probs

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "temperature": self.temperature,
            "is_fitted": self.is_fitted
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load(self, path: str | Path) -> None:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.temperature = float(data.get("temperature", 1.0))
        self.is_fitted = bool(data.get("is_fitted", True))


class MonteCarloUncertaintyEstimator:
    """
    Epistemic Uncertainty Estimation via Monte Carlo Dropout sampling.
    Estimates predictive mean, variance, standard deviation, and 95% credible/conformal intervals
    for continuous severity predictions and ordinal classification probabilities.
    Integrates multi-factor QC triage evaluation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_samples: int = 25,
        uncertainty_threshold: float = 8.0,
        conformal_calibrator: Optional[ConformalPredictionCalibrator] = None,
        temperature_scaler: Optional[TemperatureScaler] = None,
        min_det_conf: float = 0.50,
        max_conformal_width: float = 30.0
    ):
        self.model = model
        self.n_samples = n_samples
        self.uncertainty_threshold = uncertainty_threshold
        self.conformal_calibrator = conformal_calibrator
        self.temperature_scaler = temperature_scaler
        self.min_det_conf = min_det_conf
        self.max_conformal_width = max_conformal_width

    @torch.no_grad()
    def estimate_uncertainty(
        self,
        x: torch.Tensor,
        detector_confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Performs N stochastic forward passes on a single input tensor x (shape: [1, C, H, W]).
        """
        self.model.eval()

        scores_list = []
        logits_list = []

        for _ in range(self.n_samples):
            score, logits = self.model(x, enable_mc_dropout=True)
            scores_list.append(score.cpu().numpy().flatten())
            logits_list.append(logits.cpu().numpy())

        # Shape: (n_samples, 1)
        scores_arr = np.array(scores_list) # [n_samples, 1]
        logits_arr = np.array(logits_list) # [n_samples, 1, n_classes]

        mean_score = float(np.mean(scores_arr, axis=0)[0])
        std_score = float(np.std(scores_arr, axis=0)[0])
        var_score = float(np.var(scores_arr, axis=0)[0])

        raw_lower_95 = float(np.clip(mean_score - 1.96 * std_score, 0.0, 100.0))
        raw_upper_95 = float(np.clip(mean_score + 1.96 * std_score, 0.0, 100.0))

        # Conformal interval
        is_conformal = False
        if self.conformal_calibrator is not None and self.conformal_calibrator.is_calibrated:
            ci_lower, ci_upper = self.conformal_calibrator.predict_interval(mean_score, std_score)
            is_conformal = True
        else:
            ci_lower, ci_upper = raw_lower_95, raw_upper_95

        ci_lower = float(ci_lower)
        ci_upper = float(ci_upper)
        conformal_width = float(ci_upper - ci_lower)

        # Mean logits and calibrated probabilities
        mean_logits = np.mean(logits_arr, axis=0)[0] # [n_classes]
        if self.temperature_scaler is not None and self.temperature_scaler.is_fitted:
            probs = self.temperature_scaler.predict_proba(mean_logits[np.newaxis, :])[0]
        else:
            exp_l = np.exp(mean_logits - np.max(mean_logits))
            probs = exp_l / np.sum(exp_l)

        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))
        pred_class_id = int(np.argmax(probs))

        categories = ["Low", "Moderate", "High", "Critical"]
        pred_class_name = categories[pred_class_id] if pred_class_id < len(categories) else "Unknown"

        # Multi-factor QC Triage Evaluation
        qc_status, requires_triage, triage_reasons = evaluate_qc_triage(
            detector_confidence=detector_confidence,
            mc_dropout_std=std_score,
            conformal_interval_width=conformal_width,
            min_detector_confidence=self.min_det_conf,
            max_mc_std=self.uncertainty_threshold,
            max_conformal_width=self.max_conformal_width
        )

        return {
            "mean_severity_score": mean_score,
            "std_severity_score": std_score,
            "var_severity_score": var_score,
            "raw_mc_ci_95_lower": raw_lower_95,
            "raw_mc_ci_95_upper": raw_upper_95,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "conformal_interval_width": conformal_width,
            "is_conformal_calibrated": is_conformal,
            "predicted_class_id": pred_class_id,
            "predicted_class_name": pred_class_name,
            "class_probabilities": probs.tolist(),
            "predictive_entropy": entropy,
            "qc_status": qc_status,
            "requires_human_triage": requires_triage,
            "triage_reasons": triage_reasons,
            "all_mc_scores": scores_arr[:, 0].tolist()
        }

    @torch.no_grad()
    def estimate_uncertainty_batch(self, x_batch: torch.Tensor) -> Dict[str, Any]:
        """
        Performs N stochastic forward passes on a batch of tensors x_batch (shape: [B, C, H, W]).
        """
        self.model.eval()
        scores_list = []
        logits_list = []

        for _ in range(self.n_samples):
            score, logits = self.model(x_batch, enable_mc_dropout=True)
            scores_list.append(score.cpu().numpy())
            logits_list.append(logits.cpu().numpy())

        # Shape: (n_samples, batch_size)
        scores_arr = np.array(scores_list) # [n_samples, B]
        logits_arr = np.array(logits_list) # [n_samples, B, n_classes]

        mean_scores = np.mean(scores_arr, axis=0) # [B]
        std_scores = np.std(scores_arr, axis=0)   # [B]
        var_scores = np.var(scores_arr, axis=0)   # [B]

        raw_lowers = np.clip(mean_scores - 1.96 * std_scores, 0.0, 100.0)
        raw_uppers = np.clip(mean_scores + 1.96 * std_scores, 0.0, 100.0)

        if self.conformal_calibrator is not None and self.conformal_calibrator.is_calibrated:
            conf_lowers, conf_uppers = self.conformal_calibrator.predict_interval(mean_scores, std_scores)
        else:
            conf_lowers, conf_uppers = raw_lowers, raw_uppers

        conf_widths = conf_uppers - conf_lowers
        mean_logits = np.mean(logits_arr, axis=0) # [B, n_classes]

        if self.temperature_scaler is not None and self.temperature_scaler.is_fitted:
            probs = self.temperature_scaler.predict_proba(mean_logits)
        else:
            max_l = np.max(mean_logits, axis=-1, keepdims=True)
            exp_l = np.exp(mean_logits - max_l)
            probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

        pred_class_ids = np.argmax(probs, axis=-1)
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=-1)

        triage_flags = (
            (std_scores > self.uncertainty_threshold) |
            (conf_widths > self.max_conformal_width) |
            (entropy > 1.1)
        )

        return {
            "mean_severity_scores": mean_scores,
            "std_severity_scores": std_scores,
            "var_severity_scores": var_scores,
            "raw_mc_ci_95_lowers": raw_lowers,
            "raw_mc_ci_95_uppers": raw_uppers,
            "ci_95_lowers": conf_lowers,
            "ci_95_uppers": conf_uppers,
            "conformal_interval_widths": conf_widths,
            "predicted_class_ids": pred_class_ids,
            "mean_logits": mean_logits,
            "class_probabilities": probs,
            "predictive_entropies": entropy,
            "requires_human_triage": triage_flags
        }
