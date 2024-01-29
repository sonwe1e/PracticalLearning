import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class CONVNORMACT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        norm=nn.BatchNorm2d,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, groups=groups
            ),
            norm(out_channels) if norm is not None else nn.Identity(),
            activation() if activation is not None else nn.Identity(),
        )

    def forward(self, x):
        return self.m(x)


class INCEPTIONv1(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False, **kwargs):
        super().__init__()
        self.conv1x1 = CONVNORMACT(in_channels, out_channels // 4, 1, 1, 0)
        self.conv3x3 = nn.Sequential(
            CONVNORMACT(in_channels, out_channels // 4, 1, 1, 0),
            CONVNORMACT(out_channels // 4, out_channels // 4, 3, 1, 1),
        )
        self.conv5x5 = nn.Sequential(
            CONVNORMACT(in_channels, out_channels // 4, 1, 1, 0),
            CONVNORMACT(out_channels // 4, out_channels // 4, 5, 1, 2),
        )
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            CONVNORMACT(in_channels, out_channels // 4, 1, 1, 0),
        )
        self.residual = residual
        if residual:
            if in_channels != out_channels:
                self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            else:
                self.residual_conv = nn.Identity()

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x4 = self.maxpool(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x


class RESNETBOTTLENECK(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m = nn.Sequential(
            CONVNORMACT(in_channels, out_channels // 2, 1, 1, 0),
            CONVNORMACT(out_channels // 2, out_channels // 2, 3, 1, 1),
            CONVNORMACT(out_channels // 2, out_channels, 1, 1, 0, activation=None),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        return torch.relu(self.m(x) + self.shortcut(x))


class RESNEXTBOTTLENECK(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m = nn.Sequential(
            CONVNORMACT(in_channels, out_channels, 1, 1, 0),
            CONVNORMACT(out_channels, out_channels, 3, 1, 1, groups=32),
            CONVNORMACT(out_channels, out_channels, 1, 1, 0, activation=None),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        return torch.relu(self.m(x) + self.shortcut(x))


class RESNETBASICBLOCK(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m = nn.Sequential(
            CONVNORMACT(in_channels, out_channels, 3, 1, 1),
            CONVNORMACT(out_channels, out_channels, 3, 1, 1, activation=None),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        return torch.relu(self.m(x) + self.shortcut(x))


class CONVNEXTBLOCK(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m = nn.Sequential(
            CONVNORMACT(
                in_channels,
                in_channels,
                7,
                1,
                3,
                groups=in_channels,
                norm=LayerNorm,
                activation=None,
            ),
            CONVNORMACT(
                in_channels, in_channels * 4, 1, 1, 0, norm=None, activation=nn.GELU
            ),
            CONVNORMACT(
                in_channels * 4, out_channels, 1, 1, 0, norm=None, activation=None
            ),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        return torch.relu(self.m(x) + self.shortcut(x))


class MOBILENETV1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m = nn.Sequential(
            CONVNORMACT(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            CONVNORMACT(in_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.m(x)


class MOBILENETV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m = nn.Sequential(
            CONVNORMACT(in_channels, in_channels * 4, 1, 1, 0, activation=nn.ReLU6),
            CONVNORMACT(
                in_channels * 4,
                in_channels * 4,
                3,
                1,
                1,
                groups=in_channels * 4,
                activation=nn.ReLU6,
            ),
            CONVNORMACT(in_channels * 4, out_channels, 1, 1, 0, activation=None),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        return torch.relu(self.m(x) + self.shortcut(x))


class SE(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            CONVNORMACT(in_size, in_size // reduction, 1, 1, 0),
            CONVNORMACT(
                in_size // reduction, in_size, 1, 1, 0, activation=nn.Hardsigmoid
            ),
        )

    def forward(self, x):
        return x * self.se(x)


class MOBILENETV3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m = nn.Sequential(
            CONVNORMACT(in_channels, in_channels * 4, 1, 1, 0, activation=nn.Hardswish),
            CONVNORMACT(
                in_channels * 4,
                in_channels * 4,
                3,
                1,
                1,
                activation=nn.Hardswish,
                groups=in_channels * 4,
            ),
        )
        self.se = SE(in_channels * 4)
        self.n = CONVNORMACT(in_channels * 4, out_channels, 1, 1, 0, activation=None)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        return self.n(self.se(self.m(x))) + self.shortcut(x)


class selfnet(nn.Module):
    def __init__(self, m=CONVNORMACT):
        super().__init__()
        self.stem = CONVNORMACT(3, 128, 7, 2, 3)
        self.conv1 = nn.Sequential(*[m(128, 128) for _ in range(2)])
        self.conv2 = m(128, 256)
        self.conv3 = nn.Sequential(*[m(256, 256) for _ in range(2)])
        self.conv4 = m(256, 512)
        self.conv5 = nn.Sequential(*[m(512, 512) for _ in range(2)])
        self.conv6 = m(512, 512)
        self.conv7 = nn.Sequential(*[m(512, 512) for _ in range(2)])
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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = selfnet(MOBILENETV2)
    print(model)
    x = torch.randn(1, 3, 112, 112)
    y = model(x)
    print(y.shape)
