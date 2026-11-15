"""SqueezeNet features and two YOLOv3 heads; initialized without downloaded weights."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import squeezenet1_1


class SqueezeYOLOv3(nn.Module):
    def __init__(self, anchors, image_size=227):
        super().__init__()
        if anchors.shape != (2, 3, 2) or not torch.isfinite(anchors).all() or (anchors <= 0).any():
            raise ValueError("Expected six positive width/height anchors, grouped by two heads.")
        if image_size != 227:
            raise ValueError("This small experiment uses 227 x 227 inputs.")
        self.image_size = image_size
        self.register_buffer("anchors", anchors.clone())
        self.backbone = squeezenet1_1(weights=None).features
        self.coarse = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU())
        self.fine = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU())
        self.outputs = nn.ModuleList([nn.Conv2d(256, 18, 1), nn.Conv2d(128, 18, 1)])
        # Start with low object probability, since almost every grid cell is background.
        for layer in self.outputs:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
            with torch.no_grad():
                layer.bias.view(3, 6)[:, 4] = -4.5

    def freeze_stem(self):
        """Freeze the first convolution, matching the earlier experiment's intent."""
        self.backbone[0].requires_grad_(False)

    def forward(self, images):
        if images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError("Expected a batch of RGB images with shape (B, 3, 227, 227).")
        fine_features = self.backbone[:8](images)  # fire5: 256 x 28 x 28
        coarse_features = self.backbone[8:](fine_features)  # fire9: 512 x 14 x 14
        coarse = self.coarse(coarse_features)
        upsampled = F.interpolate(coarse, size=fine_features.shape[-2:], mode="nearest")
        fine = self.fine(torch.cat([upsampled, fine_features], dim=1))
        predictions = []
        for layer, features in zip(self.outputs, (coarse, fine)):
            raw = layer(features)
            batch, _, height, width = raw.shape
            predictions.append(raw.view(batch, 3, 6, height, width).permute(0, 1, 3, 4, 2))
        return predictions
