
import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, pretrained=True):
    # ResNet-18 backbone
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def freeze_backbone(model, freeze=True):
    if freeze:
        for name, p in model.named_parameters():
            if not name.startswith('fc'):
                p.requires_grad = False
    return model
