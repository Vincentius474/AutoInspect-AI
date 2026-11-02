import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """A convolutional block consisting of a convolutional layer, batch normalization, ReLU activation, and max pooling."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)

class CustomCNN(nn.Module):
    """A custom convolutional neural network for image classification."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    model = CustomCNN()
    sample = torch.randn(2, 3, 224, 224)
    output = model(sample)
    print(model)
    print("Output shape:", output.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
