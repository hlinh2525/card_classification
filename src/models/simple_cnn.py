import torch
from torch import nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )


class SimpleCNN(nn.Module):
    """CNN đơn giản cho ảnh RGB: 4 khối conv-BN-ReLU-pool -> GAP -> Linear."""

    def __init__(self, num_classes=53, channels=(32, 64, 128, 256), dropout=0.3):
        super().__init__()

        blocks = []
        in_channels = 3

        for out_channels in channels:
            blocks.append(conv_block(in_channels, out_channels))
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, images):
        return self.classifier(self.pool(self.features(images)))


if __name__ == "__main__":
    model = SimpleCNN()
    output = model(torch.randn(2, 3, 224, 224))
    print(tuple(output.shape))
