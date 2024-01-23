import torch
import torch.nn as nn


class basicblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.m(x)


class selfnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = basicblock(3, 64, 3)
        self.conv1 = basicblock(64, 64, 3)
        self.conv2 = basicblock(64, 64, 3)
        self.conv3 = basicblock(64, 128, 3)
        self.conv4 = basicblock(128, 128, 3)
        self.conv5 = basicblock(128, 256, 3)
        self.conv6 = basicblock(256, 256, 3)
        self.conv7 = basicblock(256, 512, 3)
        self.conv8 = basicblock(512, 512, 3)
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 200)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = selfnet()
    print(model)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(y.shape)
