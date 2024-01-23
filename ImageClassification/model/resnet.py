import torch.nn as nn
import torch


class resnet(nn.Module):
    def __init__(self, act=nn.ReLU()) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = act
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        # 56x56x64 -> 56x56x64
        self.layer1 = self._make_layer(64, 64, 2)
        # 56x56x64 -> 28x28x128
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        # 28x28x128 -> 14x14x256
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        # 14x14x256 -> 7x7x512
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 7x7x512 -> 1x1x512
        self.fc = nn.Linear(512, 200)

    def _make_layer(self, in_channels, out_channels, block_num, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for i in range(1, block_num):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, act=nn.ReLU()):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = act
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.act(x)
        return x


if __name__ == "__main__":
    model = resnet()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
