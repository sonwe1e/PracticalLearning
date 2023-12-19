import timm
import torch
import torch.nn as nn


class GlobalNet(nn.Module):
    def __init__(
        self, num_class, output_shape=(256, 192), channels=[64, 128, 256, 512]
    ):
        super().__init__()
        self.predict_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        for i in range(len(channels)):
            self.predict_layers.append(
                self._predict(channels[i], output_shape, num_class)
            )
            self.upsample_layers.append(
                self._upsample(channels[i], channels[i - 1])
            ) if i != 0 else self.upsample_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i], 3, 1, 1),
                    nn.BatchNorm2d(channels[i]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[i], channels[i], 1, 1, 0),
                )
            )

    def _predict(self, channel, output_shape, num_class):
        return nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_class, kernel_size=1),
            nn.Upsample(size=output_shape, mode="bilinear", align_corners=True),
        )

    def _upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

    def forward(self, feature_list):
        l = len(feature_list)
        global_output = []
        for i in range(l - 1, -1, -1):
            print(i)
            global_output.append(self.predict_layers[i](feature_list[i]))
            feature_list[i] = (
                (self.upsample_layers[i](feature_list[i]) + feature_list[i - 1])
                if i != 0
                else self.upsample_layers[i](feature_list[i])
            )
        return global_output, feature_list


if __name__ == "__main__":
    model = GlobalNet(17)
    feature_list = [
        torch.randn(1, 64, 64, 64),
        torch.randn(1, 128, 32, 32),
        torch.randn(1, 256, 16, 16),
        torch.randn(1, 512, 8, 8),
    ]
    global_output, feature_list = model(feature_list)
    for i in range(len(global_output)):
        print(global_output[i].shape)
    for i in range(len(feature_list)):
        print(feature_list[i].shape)
