import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error
)
from typing import Dict, Any

class OrdinalSeverityEvaluator:
    """
    Evaluates severity predictions considering the ordinal relationship (Low < Moderate < High < Critical).
    """

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

        # Ordinal metrics
        mae = mean_absolute_error(y_true, y_pred)
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

        # Correlations
        if len(y_true) > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            rho, _ = spearmanr(y_true, y_pred)
            tau, _ = kendalltau(y_true, y_pred)
        else:
            rho, tau = 0.0, 0.0

        cm = confusion_matrix(y_true, y_pred)

        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "precision": float(precision),
            "recall": float(recall),
            "ordinal_mae": float(mae),
            "quadratic_weighted_kappa": float(qwk),
            "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
            "kendall_tau": float(tau) if not np.isnan(tau) else 0.0,
            "confusion_matrix": cm.tolist()
        }
