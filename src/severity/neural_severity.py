import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Dict, Any

class EfficientNetV2SeverityModel(nn.Module):
    """
    Deep Neural Severity Estimation Network based on EfficientNet-V2.
    Dual-head architecture predicting both continuous severity score S in [0, 100]
    and 4-class ordinal severity grades (Low, Moderate, High, Critical).
    Supports Monte Carlo Dropout for epistemic uncertainty quantification.
    """

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_v2_s(weights=weights)

        # Extract convolutional feature backbone
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = 1280 # EfficientNet-V2-S output channels
        self.dropout_rate = dropout_rate

        # Intermediate embedding projection
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=self.dropout_rate)
        )

        # 1. Continuous Regression Head (Predicts S in [0, 100])
        self.regression_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid() # Bound output to [0, 1] then scaled by 100
        )

        # 2. Ordinal Classification Head (4 Classes)
        self.classification_head = nn.Linear(256, 4)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts spatial feature maps before pooling (used for Grad-CAM)."""
        return self.features(x)

    def forward(self, x: torch.Tensor, enable_mc_dropout: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.features(x)
        pooled = self.avgpool(feat_map)
        flat = torch.flatten(pooled, 1)

        if enable_mc_dropout:
            # Keep dropout active during inference
            flat = nn.functional.dropout(flat, p=self.dropout_rate, training=True)
            emb = self.shared_fc[0](flat)
            emb = self.shared_fc[1](emb)
            emb = nn.functional.dropout(emb, p=self.dropout_rate, training=True)
        else:
            emb = self.shared_fc(flat)

        score = self.regression_head(emb) * 100.0
        logits = self.classification_head(emb)

        return score.squeeze(-1), logits
