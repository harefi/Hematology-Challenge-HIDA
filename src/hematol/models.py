from torchvision import models
import torch.nn as nn


def build_baseline_model(num_classes: int):
    """ResNet-18 pretrained on ImageNet, FC layer resized."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
