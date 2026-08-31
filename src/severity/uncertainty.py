import torch
import numpy as np
from typing import Dict, Any, Tuple

class MonteCarloUncertaintyEstimator:
    """
    Epistemic Uncertainty Estimation via Monte Carlo Dropout sampling.
    Estimates predictive mean, variance, standard deviation, and 95% credible intervals
    for continuous severity predictions and ordinal classification probabilities.
    """

    def __init__(self, model: torch.nn.Module, n_samples: int = 30, uncertainty_threshold: float = 8.0):
        self.model = model
        self.n_samples = n_samples
        self.uncertainty_threshold = uncertainty_threshold

    @torch.no_grad()
    def estimate_uncertainty(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Performs N stochastic forward passes on input tensor x (shape: [1, C, H, W] or [B, C, H, W]).
        """
        self.model.eval()

        scores_list = []
        probs_list = []

        for _ in range(self.n_samples):
            score, logits = self.model(x, enable_mc_dropout=True)
            probs = torch.softmax(logits, dim=-1)
            scores_list.append(score.cpu().numpy())
            probs_list.append(probs.cpu().numpy())

        # Array shape: [n_samples, batch_size]
        scores_arr = np.array(scores_list)
        probs_arr = np.array(probs_list)

        # Compute empirical statistics across MC passes
        mean_score = np.mean(scores_arr, axis=0)
        std_score = np.std(scores_arr, axis=0)
        var_score = np.var(scores_arr, axis=0)

        lower_95 = np.clip(mean_score - 1.96 * std_score, 0.0, 100.0)
        upper_95 = np.clip(mean_score + 1.96 * std_score, 0.0, 100.0)

        mean_probs = np.mean(probs_arr, axis=0)
        entropy = -np.sum(mean_probs * np.log(np.clip(mean_probs, 1e-8, 1.0)), axis=-1)
        pred_class_id = np.argmax(mean_probs, axis=-1)

        categories = ["Low", "Moderate", "High", "Critical"]
        pred_class_name = categories[int(pred_class_id[0])] if hasattr(pred_class_id, "__len__") else categories[int(pred_class_id)]

        # Human triage flag: if standard deviation exceeds threshold or entropy is high
        flag_for_triage = bool(std_score[0] > self.uncertainty_threshold or entropy[0] > 1.1)

        return {
            "mean_severity_score": float(mean_score[0]) if hasattr(mean_score, "__len__") else float(mean_score),
            "std_severity_score": float(std_score[0]) if hasattr(std_score, "__len__") else float(std_score),
            "var_severity_score": float(var_score[0]) if hasattr(var_score, "__len__") else float(var_score),
            "ci_95_lower": float(lower_95[0]) if hasattr(lower_95, "__len__") else float(lower_95),
            "ci_95_upper": float(upper_95[0]) if hasattr(upper_95, "__len__") else float(upper_95),
            "predicted_class_id": int(pred_class_id[0]) if hasattr(pred_class_id, "__len__") else int(pred_class_id),
            "predicted_class_name": pred_class_name,
            "class_probabilities": mean_probs[0].tolist() if hasattr(mean_probs[0], "tolist") else mean_probs.tolist(),
            "predictive_entropy": float(entropy[0]) if hasattr(entropy, "__len__") else float(entropy),
            "requires_human_triage": flag_for_triage,
            "all_mc_scores": scores_arr[:, 0].tolist() if scores_arr.ndim > 1 else scores_arr.tolist()
        }
