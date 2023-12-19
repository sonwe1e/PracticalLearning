from NeRF_model import *
from load_llff import load_llff_data
import numpy as np
import os
import torch
from NeRF_helper import *
import copy
import time
from tqdm import tqdm
from lion_pytorch import Lion


def config_parser():
    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "--datadir", type=str, default="/media/hdd/nerf_llff_data/fern", help="输入数据目录"
    )
    parser.add_argument("--expname", type=str, default="test", help="实验名称")
    parser.add_argument("--basedir", type=str, default="./logs/", help="存储检查点和日志的位置")

    # 训练选项
    parser.add_argument("--N_iters", type=int, default=20001, help="NeRF训练次数")
    parser.add_argument("--device", type=str, default="cuda:0", help="使用的设备")
    parser.add_argument("--netdepth", type=int, default=8, help="网络层数")
    parser.add_argument("--netwidth", type=int, default=256, help="每层的通道数")
    parser.add_argument("--netdepth_fine", type=int, default=8, help="细化网络的层数")
    parser.add_argument("--netwidth_fine", type=int, default=256, help="细化网络每层的通道数")
    parser.add_argument("--N_rand", type=int, default=4096, help="批处理大小(每个梯度步骤的随机光线数)")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="学习率")
    parser.add_argument("--chunk", type=int, default=32768, help="并行处理的光线数")
    parser.add_argument("--netchunk", type=int, default=1024 * 64, help="并行通过网络发送的点数")
    parser.add_argument("--ft_path", type=str, default=None, help="加载网络权重的文件夹/文件路径")

    # 渲染选项
    parser.add_argument("--N_samples", type=int, default=64, help="每条光线的粗略样本数")
    parser.add_argument("--N_importance", type=int, default=0, help="每条光线的额外精细样本数")
    parser.add_argument("--perturb", type=bool, default=False, help="False表示无抖动")
    parser.add_argument("--use_viewdirs", default=True, help="使用完整的5D输入而不是3D")
    parser.add_argument("--position_encode", type=int, default=0, help="设置0表示默认位置编码")
    parser.add_argument("--multires", type=int, default=10, help="位置编码(3D位置)的最大频率的对数")
    parser.add_argument("--multires_views", type=int, default=4, help="位置编码的最大频率的对数")
    parser.add_argument("--raw_noise_std", type=float, default=0.0, help="sigma_a的噪声")

    parser.add_argument("--render_only", default=False, help="渲染render_poses路径")
    parser.add_argument("--render_test", action="store_true", help="渲染测试集")
    parser.add_argument("--render_factor", type=int, default=0, help="加快渲染速度的下采样因子 4|8")

    # 训练选项
    parser.add_argument("--precrop_iters", type=int, default=0, help="训练中心裁剪的步骤数")
    parser.add_argument("--precrop_frac", type=float, default=0.5, help="中心裁剪所占图片的比例")

    # 数据集选项
    parser.add_argument("--dataset_type", default="llff", help="blender|deepvoxels")
    parser.add_argument("--testskip", type=int, default=8, help="从数据集中加载1/N的图像")
    ## deepvoxels标志
    parser.add_argument("--shape", type=str, default="greek", help="armchair|cube|vase")

    ## blender标志
    parser.add_argument("--white_bkgd", action="store_true", help="在白色背景上渲染合成数据")
    parser.add_argument("--half_res", action="store_true", help="以一半分辨率加载blender合成数据")

    ## llff标志
    parser.add_argument("--factor", type=int, default=4, help="LLFF图像的降采样因子")
    parser.add_argument("--no_ndc", action="store_true", help="不使用标准化设备坐标(设置用于非前向场景)")
    parser.add_argument("--lindisp", action="store_true", help="在视差而不是深度上线性采样")
    parser.add_argument("--spherify", action="store_true", help="设置用于球形360场景")
    parser.add_argument("--llffhold", type=int, default=8, help="将每1/N张图像作为测试集")

    # 日志/保存选项
    parser.add_argument("--i_print", type=int, default=100, help="控制台打印输出和度量日志的频率")
    parser.add_argument("--i_weights", type=int, default=2000, help="权重检查点保存的频率")
    parser.add_argument("--i_video", type=int, default=10000, help="视频保存的频率")

    return parser


def main():
    K = None
    parser = config_parser()
    args = parser.parse_args()

    # 选择数据集
    if args.dataset_type == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir,
            args.factor,
            recenter=True,
            bd_factor=0.75,
            spherify=args.spherify,
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:  # 如果设置了 llffhold 通过每 1/N 张图像作为测试集
            print("Auto LLFF holdout,", args.llffhold)
            i_test = np.arange(images.shape[0])[:: args.llffhold]

        i_val = i_test
        i_train = np.array(
            [
                i
                for i in np.arange(int(images.shape[0]))
                if (i not in i_test and i not in i_val)
            ]
        )

        if args.no_ndc:  # 如果不使用标准化设备坐标 Normalize Device Coordinates
            near = np.ndarray.min(bds) * 0.9
            far = np.ndarray.max(bds) * 1.0
        else:
            near = 0.0
            far = 1.0
        print("NEAR FAR", near, far)

    elif args.dataset_type == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip
        )
        print("Loaded blender", images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.0
        far = 6.0

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == "LINEMOD":
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(
            args.datadir, args.half_res, args.testskip
        )
        print(f"Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}")
        print(f"[CHECK HERE] near: {near}, far: {far}.")
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == "deepvoxels":
        images, poses, render_poses, hwf, i_split = load_dv_data(
            scene=args.shape, basedir=args.datadir, testskip=args.testskip
        )

        print("Loaded deepvoxels", images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.0
        far = hemi_R + 1.0

    else:
        print("Unknown dataset type", args.dataset_type, "exiting")
        return

    # 将内参矩阵(intrinsics)转换为正确的数据类型
    H, W, focal = hwf  # 从 hwf 中提取高度(H),宽度(W)和焦距(focal)
    H, W = int(H), int(W)  # 确保高度和宽度是整数类型
    hwf = [H, W, focal]  # 重新组合 hwf
    # 相机内参
    K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])
    if args.render_test:
        render_poses = np.array(poses[i_test])

    # 创建日志目录并复制配置文件
    os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)  # 创建目录
    # 获取位置编码函数
    embed_fn, input_ch = get_embedder(args.multires, args.position_encode)
    embeddirs_fn, input_ch_views = (
        get_embedder(args.multires_views, args.position_encode)
        if args.use_viewdirs
        else (None, 0)
    )
    output_ch = 5 if args.N_importance > 0 else 4
    # 定义 NeRF 模型
    model = NeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=[4],
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs,
    ).to(args.device)
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(
            D=args.netdepth_fine,
            W=args.netwidth_fine,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=[4],
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
        ).to(args.device)
    # 定义优化器
    model_param = (
        list(model.parameters())
        if model_fine is None
        else list(model.parameters()) + list(model_fine.parameters())
    )
    model_fn = {
        "network_fn": model,
        "network_fine": model_fine,
        "embed_fn": embed_fn,
        "embeddirs_fn": embeddirs_fn,
        "netchunk": args.netchunk,
    }
    render_args = {
        "lindisp": args.lindisp,
        "perturb": args.perturb,
        "N_samples": args.N_samples,
        "N_importance": args.N_importance,
        "white_bkgd": args.white_bkgd,
        "raw_noise_std": args.raw_noise_std,
        "pytest": False,
    }
    optimizer = Lion(model_param, lr=args.learning_rate)
    start = 0
    # 加载预训练模型
    if args.ft_path is not None:
        if os.path.isfile(args.ft_path):
            ckpt_path = args.ft_path
        elif os.path.isdir(args.ft_path):
            ckpt_path = sorted(glob.glob(os.path.join(args.ft_path, "*.tar")))[-1]
        ckpt = torch.load(ckpt_path)
        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        model_fn["network_fn"].load_state_dict(ckpt["network_fn_state_dict"])
        if model_fn["network_fine"] is not None:
            model_fn["network_fine"].load_state_dict(ckpt["network_fine_state_dict"])
        print("Loaded checkpoint from", ckpt_path)

    global_step = start  # 初始化全局步数
    render_poses = torch.Tensor(render_poses).to(args.device)

    # 如果仅从训练好的模型渲染输出,则提前结束
    if args.render_only:
        print("Render only")
        with torch.no_grad():
            if args.render_test:
                images = images[i_test]
            else:
                images = None

            testsavedir = os.path.join(args.basedir, args.expname, f"render_{start}")
            os.makedirs(testsavedir, exist_ok=True)
            render_args["perturb"] = False
            render_args["raw_noise_std"] = 0.0
            H, W, focal = hwf  # 提取图像高度、宽度和焦距

            if args.render_factor != 0:
                # 如果指定了渲染因子，进行降采样以加快渲染速度
                H = H // args.render_factor
                W = W // args.render_factor
                focal = focal / args.render_factor

            rgbs = []  # 存储渲染出的RGB图像
            disps = []  # 存储渲染出的视差图

            for i, c2w in enumerate(tqdm(render_poses)):
                # 调用render函数进行渲染
                rgb, disp, acc, _ = render(
                    H,
                    W,
                    K,
                    c2w=c2w[:3, :4],
                    chunk=args.chunk,
                    model_fn=model_fn,
                    render_args=render_args,
                )
                rgbs.append(rgb.cpu().numpy())  # 保存RGB图像
                disps.append(disp.cpu().numpy())  # 保存视差图

                # 如果指定了保存目录，则将RGB图像保存为PNG文件
                if testsavedir is not None:
                    imageio.imwrite(
                        os.path.join(testsavedir, f"{i:03d}.png"),
                        to8b(rgbs[-1]),
                    )

            # 将所有RGB图像和视差图堆叠起来
            rgbs = np.stack(rgbs, 0)
            disps = np.stack(disps, 0)

            print("Done rendering", testsavedir)
            # 将渲染结果保存为视频
            imageio.mimwrite(
                os.path.join(testsavedir, "video.mp4"), to8b(rgbs), fps=30, quality=8
            )

            return  # 结束函数执行

    N_rand = args.N_rand  # 随机光线数
    print("Getting rays and RGB")
    rays = np.stack([get_rays_np(H, W, K, p) for p in poses], 0)  # [N, ro+rd, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.astype(np.float32)
    np.random.shuffle(rays_rgb)
    print("Done Loading")

    i_batch = 0
    images = torch.Tensor(images).to(args.device)
    rays_rgb = torch.Tensor(rays_rgb).to(args.device)
    poses = torch.Tensor(poses).to(args.device)

    print("TRAIN views are", i_train)
    print("TEST views are ", i_test)
    print("VAL views are  ", i_val)

    start = start + 1
    for i in range(start, args.N_iters):
        batch = rays_rgb[i_batch : i_batch + N_rand]  # [B, 3, 3]
        batch = torch.transpose(batch, 0, 1)  # [3, B, 3]
        batch_rays, target_s = batch[:2], batch[2]  # [2, B, 3], [B, 3]

        i_batch += N_rand
        # 每次打乱光线顺序 重新开始
        if i_batch >= rays_rgb.shape[0]:
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            i_batch = 0

        rgb, disp, acc, extras = render(
            H=H,
            W=W,
            K=K,
            near=near,
            far=far,
            chunk=args.chunk,
            rays=batch_rays,
            model_fn=model_fn,
            render_args=render_args,
            use_viewdirs=args.use_viewdirs,
        )

        optimizer.zero_grad()
        loss = img2mse(rgb, target_s)
        psnr = mse2psnr(loss)

        if "rgb0" in extras:
            img_loss0 = img2mse(extras["rgb0"], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        if i % args.i_weights == 0:
            path = os.path.join(args.basedir, args.expname, f"{i:06d}.ckpt")
            torch.save(
                {
                    "global_step": global_step,
                    "network_fn_state_dict": model_fn["network_fn"].state_dict(),
                    "network_fine_state_dict": model_fn["network_fine"].state_dict()
                    if model_fn["network_fine"] is not None
                    else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
        if i % args.i_video == 0 and i > 0:
            with torch.no_grad():
                rgbs, disps = render_path(
                    render_poses,
                    hwf,
                    K,
                    args.chunk,
                    model_fn,
                    render_args,
                )
            print("Done, saving", rgbs.shape, disps.shape)
            moviebase = os.path.join(
                args.basedir, args.expname, "{}_spiral_{:06d}_".format(args.expname, i)
            )
            imageio.mimwrite(moviebase + "rgb.mp4", to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(
                moviebase + "disp.mp4", to8b(disps / np.max(disps)), fps=30, quality=8
            )

        if i % args.i_print == 0:
            print(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        global_step += 1


if __name__ == "__main__":
    main()
