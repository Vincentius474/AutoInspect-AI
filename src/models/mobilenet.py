import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

def create_mobilenet(num_classes=3, pretrained=True):
    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
        if pretrained else None
    )

    model = mobilenet_v3_small(weights=weights)

    # Freeze the feature extractor initially.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Replace the original ImageNet classifier.
    input_features = model.classifier[3].in_features

    model.classifier[3] = nn.Linear(
        input_features,
        num_classes
    )

    return model

if __name__ == "__main__":

    import torch

    model = create_mobilenet()
    sample = torch.randn(2, 3, 224, 224)
    output = model(sample)

    print("Output shape:", output.shape)
    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )
