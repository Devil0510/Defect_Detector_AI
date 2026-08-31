import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
import xgboost as xgb

class SeverityMLModelTrainer:
    """
    Feature-based machine learning trainer for steel surface defect severity estimation.
    Supports Random Forest, Gradient Boosting (XGBoost), and Ridge Ordinal Classifier.
    """
    
    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._init_model()

    def _init_model(self):
        if self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.random_state)
        elif self.model_type == "xgboost":
            return xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=self.random_state)
        elif self.model_type == "ridge":
            return RidgeClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            df = self.model.decision_function(X)
            # Softmax conversion
            exp_df = np.exp(df - np.max(df, axis=-1, keepdims=True))
            return exp_df / np.sum(exp_df, axis=-1, keepdims=True)
        else:
            # Fallback one-hot matrix
            preds = self.predict(X)
            n_classes = len(np.unique(preds))
            one_hot = np.zeros((len(preds), n_classes))
            for i, p in enumerate(preds):
                one_hot[i, p] = 1.0
            return one_hot

    def save(self, filepath: str | Path):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: str | Path):
        path = Path(filepath)
        with open(path, "rb") as f:
            self.model = pickle.load(f)
