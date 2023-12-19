import numpy as np
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
import os
import imageio

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]).to(x.device))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def get_rays(H, W, K, c2w):
    # 计算每个像素对应的光线的起点和方向
    # H, W 分别是图像的高度和宽度，K 是内参矩阵，c2w 是相机到世界坐标系的变换矩阵

    # 使用torch.meshgrid生成网格的横纵坐标
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H)
    )  # PyTorch的meshgrid使用'ij'索引
    i = i.t()  # 转置以匹配图像的坐标系统
    j = j.t()
    # 计算每个像素点对应的光线方向
    dirs = torch.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1
    )

    # 将光线方向从相机坐标系转换到世界坐标系
    rays_d = torch.sum(
        dirs[..., np.newaxis, :].to(c2w.device) * c2w[:3, :3], -1
    )  # 点积，等同于 [c2w.dot(dir) for dir in dirs]

    # 将相机坐标系的原点（相机位置）转换到世界坐标系，这是所有光线的起点
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    # 返回光线的起点和方向
    return rays_o, rays_d


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    # 将原始预测转换为透明度
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)

    # 计算相邻采样点之间的距离
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # 为最后一个采样点添加无限远的距离
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists.device)],
        -1,
    )

    # 考虑光线方向的长度缩放
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # 使用sigmoid函数激活以得到RGB颜色
    rgb = torch.sigmoid(raw[..., :3])
    noise = 0.0
    # 如果指定了噪声标准差，则添加噪声
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # 如果是测试模式，则使用固定的随机数
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # 计算透明度
    alpha = raw2alpha(raw[..., 3] + noise, dists)
    # 计算沿光线的累积权重
    weights = (
        alpha
        * torch.cumprod(
            torch.cat(
                [
                    torch.ones((alpha.shape[0], 1), device=alpha.device),
                    1.0 - alpha + 1e-10,
                ],
                -1,
            ),
            -1,
        )[:, :-1]
    )
    # 计算RGB颜色映射
    rgb_map = torch.sum(weights[..., None] * rgb, -2)

    # 计算深度映射和视差映射
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
    )
    # 计算累积不透明度
    acc_map = torch.sum(weights, -1)

    # 如果指定了白色背景，则添加到RGB映射中
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    # 返回RGB颜色映射、视差映射、累积不透明度、权重和深度映射
    return rgb_map, disp_map, acc_map, weights, depth_map


def get_rays_np(H, W, K, c2w):
    # 计算从相机参数和相机到世界坐标系的变换中得到的光线
    # H, W 是图像的高度和宽度，K 是内参矩阵，c2w 是相机到世界坐标系的变换矩阵

    # 生成像素的网格坐标
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    # 计算光线的方向
    dirs = np.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1
    )

    # 将光线方向从相机坐标系旋转到世界坐标系
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # 点积运算

    # 将相机坐标系的原点平移到世界坐标系，这是所有光线的起点
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    # 返回光线的起点和方向
    return rays_o, rays_d


def render(
    H,
    W,
    K,
    c2w=None,
    near=0.0,
    far=1.0,
    chunk=1024 * 32,
    rays=None,
    model_fn=None,
    render_args=None,
    use_viewdirs=True,
):
    """渲染光线"""
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        # 如果使用视角方向，提供光线方向作为输入
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    # 获取光线方向的形状
    sh = rays_d.shape  # [..., 3]
    rays_o, rays_d = ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)

    # 创建光线批次
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    # 设置近平面和远平面
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
        rays_d[..., :1]
    )
    # 合并光线的各个组成部分
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    all_ret = {}
    for i in range(0, rays.shape[0], chunk):
        ret = render_rays(rays[i : i + chunk], model_fn, render_args)
        # 将当前批次的渲染结果存储到 all_ret 字典中
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    # 将所有批次的结果合并
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # 提取关键结果
    k_extract = ["rgb_map", "disp_map", "acc_map"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # 将光线原点移动到近平面
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # 进行投影转换到NDC空间
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (
        -1.0
        / (W / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        -1.0
        / (H / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    # 将计算得到的NDC坐标重新组合为光线的原点和方向
    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    # 返回转换后的光线原点和方向
    return rays_o, rays_d


def render_path(
    render_poses,
    hwf,
    K,
    chunk,
    model_fn,
    render_args,
    savedir=None,
    render_factor=0,
):
    # 根据给定的相机姿态渲染图像序列
    H, W, focal = hwf  # 提取图像高度、宽度和焦距

    if render_factor != 0:
        # 如果指定了渲染因子，进行降采样以加快渲染速度
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []  # 存储渲染出的RGB图像
    disps = []  # 存储渲染出的视差图

    for i, c2w in enumerate(tqdm(render_poses)):
        # 调用render函数进行渲染
        rgb, disp, acc, _ = render(
            H,
            W,
            K,
            c2w=c2w[:3, :4],
            chunk=chunk,
            model_fn=model_fn,
            render_args=render_args,
        )
        rgbs.append(rgb.cpu().numpy())  # 保存RGB图像
        disps.append(disp.cpu().numpy())  # 保存视差图

        # 如果指定了保存目录，则将RGB图像保存为PNG文件
        if savedir is not None:
            imageio.imwrite(
                os.path.join(savedir, "{:03d}.png".format(i)), to8b(rgbs[-1])
            )

    # 将所有RGB图像和视差图堆叠起来
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def render_rays(ray_batch, model_fn, render_args):
    """体积渲染函数"""
    N_rays = ray_batch.shape[0]  # 光线数量
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # 每条光线的原点和方向
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    near, far = ray_batch[..., 6].unsqueeze(-1), ray_batch[..., 7].unsqueeze(-1)

    # 根据光线数量和样本数量生成采样点
    t_vals = torch.linspace(
        0.0, 1.0, steps=render_args["N_samples"], device=near.device
    )
    # 根据是否线性分布确定z值
    if not render_args["lindisp"]:
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))
    z_vals = z_vals.expand([N_rays, render_args["N_samples"]])

    # 如果有扰动，则对采样点添加随机扰动
    if render_args["perturb"]:
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)
        if render_args["pytest"]:
            np.random.seed(0)
            t_rand = torch.Tensor(np.random.rand(*list(z_vals.shape))).to(z_vals.device)

        z_vals = lower + (upper - lower) * t_rand
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    embedded = model_fn["embed_fn"](pts.reshape(-1, pts.shape[-1]))
    if viewdirs is not None:  # 对视角特征进行编码
        embedded_dirs = model_fn["embeddirs_fn"](
            viewdirs.repeat_interleave(pts.shape[1], dim=0)
        )
        embedded = torch.cat([embedded, embedded_dirs], -1)
    outputs = torch.cat(
        [
            model_fn["network_fn"](embedded[i : i + model_fn["netchunk"]])
            for i in range(0, embedded.shape[0], model_fn["netchunk"])
        ],
        0,
    )
    raw = torch.reshape(outputs, list(pts.shape[:-1]) + [outputs.shape[-1]])
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw,
        z_vals,
        rays_d,
        render_args["raw_noise_std"],
        render_args["white_bkgd"],
        pytest=render_args["pytest"],
    )

    # 如果设置了重要性采样，则进行进一步采样和渲染
    if render_args["N_importance"] > 0:
        # 保存粗渲染的结果
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # 计算重要性采样点
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            render_args["N_importance"],
            det=(render_args["perturb"] == False),
            pytest=render_args["pytest"],
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples + N_importance, 3]

        embedded = model_fn["embed_fn"](pts.reshape(-1, pts.shape[-1]))

        if viewdirs is not None:  # 对视角特征进行编码
            embedded_dirs = model_fn["embeddirs_fn"](
                viewdirs.repeat_interleave(pts.shape[1], dim=0)
            )
            embedded = torch.cat([embedded, embedded_dirs], -1)
        outputs = torch.cat(
            [
                model_fn["network_fn"](embedded[i : i + model_fn["netchunk"]])
                if model_fn["network_fine"] is None
                else model_fn["network_fine"](embedded[i : i + model_fn["netchunk"]])
                for i in range(0, embedded.shape[0], model_fn["netchunk"])
            ],
            0,
        )
        raw = torch.reshape(outputs, list(pts.shape[:-1]) + [outputs.shape[-1]])

        # 从结果中计算RGB颜色、视差图、不透明度等
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw,
            z_vals,
            rays_d,
            render_args["raw_noise_std"],
            render_args["white_bkgd"],
            pytest=render_args["pytest"],
        )

    # 准备返回结果
    ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}
    # 如果进行了重要性采样，则添加粗渲染结果和样本标准差
    if render_args["N_importance"] > 0:
        ret["rgb0"] = rgb_map_0
        ret["disp0"] = disp_map_0
        ret["acc0"] = acc_map_0
        ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    # 检查数值错误
    for k in ret:
        if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0.0, 1.0, N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
