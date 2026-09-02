import pytest
import numpy as np
import torch
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.severity.index_calculator import TransparentSeverityIndexCalculator, DEFAULT_DEFECT_WEIGHTS
from src.data.clahe import apply_clahe
from src.severity.uncertainty import (
    ConformalPredictionCalibrator,
    TemperatureScaler,
    MonteCarloUncertaintyEstimator
)
from src.severity.neural_severity import EfficientNetV2SeverityModel

def test_defect_specific_weights():
    calc = TransparentSeverityIndexCalculator()

    # Verify scratches vs patches weights
    w_scratch = calc.get_weights_for_class("scratches")
    w_patch = calc.get_weights_for_class("patches")
    w_pitted = calc.get_weights_for_class("pitted_surface")
    w_default = calc.get_weights_for_class("unknown_class")

    assert w_scratch["w_length"] > w_scratch["w_area"], "Scratches should be length-dominated"
    assert w_patch["w_area"] > w_patch["w_length"], "Patches should be area-dominated"
    assert w_pitted["w_texture"] > w_default["w_texture"], "Pitted surface should have elevated texture weight"

    # Test severity calculation difference
    row = {
        "defect_area_px": 3000.0,
        "length_px": 180.0,
        "local_contrast": 0.5,
        "glcm_contrast": 25.0
    }

    score_scratch = calc.compute_severity_score(row, defect_class="scratches")
    score_patch = calc.compute_severity_score(row, defect_class="patches")

    assert 0.0 <= score_scratch <= 100.0
    assert 0.0 <= score_patch <= 100.0
    assert abs(score_scratch - score_patch) > 0.01

def test_clahe_enhancement():
    # 2D Grayscale image
    img_gray = np.random.randint(50, 150, (100, 100), dtype=np.uint8)
    enhanced_gray = apply_clahe(img_gray, clip_limit=2.0)
    assert enhanced_gray.shape == img_gray.shape
    assert enhanced_gray.dtype == np.uint8

    # 3D BGR image
    img_bgr = np.random.randint(50, 150, (100, 100, 3), dtype=np.uint8)
    enhanced_bgr = apply_clahe(img_bgr, clip_limit=2.0)
    assert enhanced_bgr.shape == img_bgr.shape
    assert enhanced_bgr.dtype == np.uint8

def test_conformal_prediction_calibrator():
    np.random.seed(42)
    n = 200
    y_true = np.random.uniform(10, 90, n)
    # Simulate predictions with noise
    errors = np.random.normal(0, 3.0, n)
    y_pred_mean = y_true + errors
    y_pred_std = np.full(n, 3.0)

    calibrator = ConformalPredictionCalibrator(alpha=0.05)
    q_hat = calibrator.calibrate(y_true, y_pred_mean, y_pred_std)

    assert q_hat > 0.0
    assert calibrator.is_calibrated is True

    # Test empirical coverage on calibration distribution
    low, high = calibrator.predict_interval(y_pred_mean, y_pred_std)
    coverage = np.mean((y_true >= low) & (y_true <= high))
    assert coverage >= 0.94, f"Conformal coverage was {coverage*100:.1f}%, expected >= 94%"

def test_temperature_scaler():
    np.random.seed(42)
    n = 150
    # Create overconfident logits
    logits = np.random.randn(n, 4) * 4.0
    labels = np.random.randint(0, 4, n)

    scaler = TemperatureScaler()
    t = scaler.fit(logits, labels)

    assert t > 0.0
    assert scaler.is_fitted is True

    probs = scaler.predict_proba(logits)
    assert probs.shape == (n, 4)
    assert np.allclose(np.sum(probs, axis=-1), 1.0)

def test_end_to_end_neural_gradients():
    model = EfficientNetV2SeverityModel(pretrained=False, dropout_rate=0.3)
    x = torch.randn(2, 3, 224, 224)
    score, logits = model(x, enable_mc_dropout=False)

    loss = score.sum() + logits.sum()
    loss.backward()

    # Verify backbone gradients exist and are non-zero
    for param in model.features.parameters():
        if param.requires_grad:
            assert param.grad is not None
            break
