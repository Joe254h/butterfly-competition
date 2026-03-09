
import torch.nn as nn
import torchvision.models as models

def build_model(num_classes: int):
    model = models.efficientnet_b0(weights="DEFAULT")
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
