from torch import nn
import torch


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = nn.Conv2d(inp_dim, out_dim // 2, 1)
        self.bn2 = nn.BatchNorm2d(out_dim // 2)
        self.conv2 = nn.Conv2d(out_dim // 2, out_dim // 2, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(out_dim // 2)
        self.conv3 = nn.Conv2d(out_dim // 2, out_dim, 1)
        self.skip_layer = nn.Conv2d(inp_dim, out_dim, 1)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class HourGlass(nn.Module):
    def __init__(self, n, f, increase=0):
        super(HourGlass, self).__init__()
        nf = f + increase
        self.residual_layer = Residual(f, f)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv1 = Residual(f, nf)
        if n > 1:
            self.mid = HourGlass(n - 1, nf)
        else:
            self.mid = Residual(nf, nf)
        self.conv2 = Residual(nf, f)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        residual = self.residual_layer(x)
        x = self.pool1(x)
        x = self.conv1(x)
        x = self.mid(x)
        x = self.conv2(x)
        x = self.up(x)
        return residual + x


class HourGlassNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, increase=0):
        super(HourGlassNet, self).__init__()

        self.nstack = nstack
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            Residual(64, 128),
            nn.MaxPool2d(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim),
        )

        self.hgs = nn.ModuleList(
            [
                nn.Sequential(
                    HourGlass(4, inp_dim, increase),
                )
                for i in range(nstack)
            ]
        )

        self.features = nn.ModuleList(
            [
                nn.Sequential(
                    Residual(inp_dim, inp_dim),
                    nn.Conv2d(inp_dim, inp_dim, 1),
                )
                for i in range(nstack)
            ]
        )

        self.outs = nn.ModuleList(
            [nn.Conv2d(inp_dim, oup_dim, 1) for i in range(nstack)]
        )
        self.merge_features = nn.ModuleList(
            [nn.Conv2d(inp_dim, inp_dim, 1) for i in range(nstack - 1)]
        )
        self.merge_preds = nn.ModuleList(
            [nn.Conv2d(oup_dim, inp_dim, 1) for i in range(nstack - 1)]
        )

    def forward(self, imgs):
        x = self.stem(imgs)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)


if __name__ == "__main__":
    model = HourGlassNet(4, 256, 17)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
