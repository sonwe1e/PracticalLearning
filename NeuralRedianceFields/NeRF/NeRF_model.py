import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


class Embedder:
    def __init__(self, max_freq_log2, num_freqs):
        embed_fns = [nn.Identity()]  # 初始化嵌入函数列表
        out_dim = 3
        freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, steps=num_freqs)

        # 对于每个频率带和周期函数，创建一个新的嵌入函数
        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += 3  # 更新输出维度

        # 保存嵌入函数列表和输出维度
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat(
            [fn(inputs) for fn in self.embed_fns], -1
        )  # 对每个嵌入函数应用输入，并在最后一个维度上合并结果


def get_embedder(multires, position_encode=0):
    # 定义一个函数来获取嵌入器
    if position_encode == -1:
        return nn.Identity(), 3

    # 创建嵌入器对象
    embedder = Embedder(multires - 1, multires)
    embed_fn = lambda x, embedder_class=embedder: embedder_class.embed(x)
    # 返回一个位置编码的函数以及输出维度
    return embed_fn, embedder.out_dim


class NeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
    ):
        """NeRF模型初始化"""
        super(NeRF, self).__init__()
        # 初始化网络参数
        self.D = D  # 网络深度
        self.W = W  # 每一层的宽度
        self.input_ch = input_ch  # 输入通道数
        self.input_ch_views = input_ch_views  # 视图输入通道数
        self.skips = skips  # 跳过连接的层
        self.use_viewdirs = use_viewdirs  # 是否使用视角方向

        # 构建点特征层
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
                for i in range(D - 1)
            ]
        )

        # 官方代码实现
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)  # 特征线性层
            self.alpha_linear = nn.Linear(W, 1)  # 透明度线性层
            self.rgb_linear = nn.Linear(W // 2, 3)  # RGB线性层
        else:
            self.output_linear = nn.Linear(W, output_ch)  # 输出线性层

    def forward(self, x):
        # 前向传播
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)  # 计算透明度
            feature = self.feature_linear(h)  # 提取特征
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)  # 计算RGB值
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)  # 计算输出

        return outputs
