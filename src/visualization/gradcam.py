import cv2
import torch
import numpy as np
from typing import Tuple

class SeverityGradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for Continuous Severity Models.
    Computes spatial activation heatmaps indicating which pixel regions contributed most
    to the predicted defect severity score.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.model.eval()
        # Default to the final convolutional block of features in EfficientNet-V2
        self.target_layer = target_layer if target_layer is not None else self.model.features[-1]
        
        self.gradients = None
        self.activations = None
        
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, original_image_shape: Tuple[int, int] = (200, 200)) -> np.ndarray:
        """
        Generates a normalized [0, 1] 2D Grad-CAM heatmap for the input tensor.
        """
        self.model.zero_grad()
        
        # Forward pass (without MC dropout for deterministic gradients)
        score, _ = self.model(input_tensor, enable_mc_dropout=False)
        
        # Backward pass on the continuous severity score
        score.backward(torch.ones_like(score), retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros(original_image_shape, dtype=np.float32)

        # Global average pooling on gradients to obtain channel importance weights alpha_k
        alpha = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        weighted_act = torch.sum(alpha * self.activations, dim=1, keepdim=True)

        # Apply ReLU to retain only positive contributions to severity
        cam = torch.relu(weighted_act).squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if np.max(cam) > np.min(cam):
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        else:
            cam = np.zeros_like(cam)

        # Resize heatmap to match target image dimensions
        h, w = original_image_shape[:2]
        resized_cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

        return resized_cam

    @staticmethod
    def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlays the 2D heatmap on the input image using false-color alpha blending.
        """
        if len(image.shape) == 2:
            base_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            base_img = image.copy()

        # Convert normalized [0, 1] heatmap to 8-bit [0, 255]
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
        color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

        # Alpha blend overlay
        overlay = cv2.addWeighted(base_img, 1.0 - alpha, color_heatmap, alpha, 0)
        return overlay
