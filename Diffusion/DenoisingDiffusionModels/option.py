import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=413)
    parser.add_argument("--save_wandb", type=bool, default=True)
    parser.add_argument("--project", type=str, default="Diffusion")

    # models
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--num_res", type=int, default=[1, 2, 4])
    parser.add_argument("--num_channels", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--scale", type=int, default=1)

    # dataset
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/sonwe1e/WorkStation/Dataset/CelebA/CelebA_Croped_Resized_32",
    )
    parser.add_argument("--image_size", type=int, default=32)

    # training setups
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.03)
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5)
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--T", type=int, default=250)
    parser.add_argument("--loss_type", type=str, default="l2")
    parser.add_argument("--pct_start", type=float, default=0.08)
    parser.add_argument("--beta1", type=float, default=1e-4)
    parser.add_argument("--betaT", type=float, default=1e-2)
    # parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--gradient_clip", type=float, default=1.0)

    # experiment
    parser.add_argument("--exp_name", type=str, default="is_32+s_1+n_3")
    parser.add_argument("--val_check", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=5)

    return parser.parse_args("")


def get_option():
    opt = parse_args()
    return opt
