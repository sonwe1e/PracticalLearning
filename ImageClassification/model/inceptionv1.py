import torch
import torch.nn as nn


class inceptionv1_block(nn.Module):
    def __init__(self, in_channels, c1, c31, c32, c51, c52, cpool):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(in_channels, c31, kernel_size=1)
        self.conv3x3 = nn.Conv2d(c31, c32, kernel_size=3, padding=1)
        self.conv1x1_5 = nn.Conv2d(in_channels, c51, kernel_size=1)
        self.conv5x5 = nn.Conv2d(c51, c52, kernel_size=5, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1x1_7 = nn.Conv2d(in_channels, cpool, kernel_size=1)
        self.a = nn.ReLU()

    def forward(self, x):
        x1 = self.a(self.conv1x1(x))
        x2 = self.a(self.conv3x3(self.a(self.conv1x1_3(x))))
        x3 = self.a(self.conv5x5(self.a(self.conv1x1_5(x))))
        x4 = self.a(self.conv1x1_7(self.maxpool(x)))
        return torch.cat([x1, x2, x3, x4], dim=1)


class inceptionv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(192)
        self.block3a = inceptionv1_block(192, 64, 96, 128, 16, 32, 32)
        self.block3b = inceptionv1_block(256, 128, 128, 192, 32, 96, 64)
        self.block4a = inceptionv1_block(480, 192, 96, 208, 16, 48, 64)
        self.block4b = inceptionv1_block(512, 160, 112, 224, 24, 64, 64)
        self.block4c = inceptionv1_block(512, 128, 128, 256, 24, 64, 64)
        self.block4d = inceptionv1_block(512, 112, 144, 288, 32, 64, 64)
        self.block4e = inceptionv1_block(528, 256, 160, 320, 32, 128, 128)
        self.block5a = inceptionv1_block(832, 256, 160, 320, 32, 128, 128)
        self.block5b = inceptionv1_block(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(1024, 200)
        self.dp = nn.Dropout(0.4)
        self.a = nn.ReLU()

    def forward(self, x):
        x = self.a(self.conv1(x))
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.a(self.conv2(x))
        x = self.a(self.conv3(x))
        x = self.bn2(x)
        x = self.maxpool(x)
        x = self.block3a(x)
        x = self.block3b(x)
        x = self.maxpool(x)
        x = self.block4a(x)
        x = self.block4b(x)
        x = self.block4c(x)
        x = self.block4d(x)
        x = self.block4e(x)
        x = self.maxpool(x)
        x = self.block5a(x)
        x = self.block5b(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.dp(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = inceptionv1()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
