import json
import torch
from option import get_option
import shutil
import os
from pl_tool import *
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from dpm import *
import torch
import yaml
import wandb


torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    opt = get_option()
    """定义网络"""
    from Unet import Unet

    Unet = Unet(3, opt.num_res, opt.num_channels, opt.n, opt.scale)
    diffusion = GaussianDiffusion(Unet, opt.beta1, opt.betaT, opt.T)
    # Define dataloader
    dataloader = prepare_data(opt)

    # Training Init
    pl.seed_everything(opt.seed)
    wandb_logger = WandbLogger(
        project=opt.project,
        name=opt.exp_name,
        offline=not opt.save_wandb,
        config=opt,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=opt.epochs,
        precision="bf16-mixed",
        default_root_dir="./",
        deterministic=False,
        logger=wandb_logger,
        val_check_interval=opt.val_check,
        log_every_n_steps=5,
        # accumulate_grad_batches=32,
        gradient_clip_val=opt.gradient_clip,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=f"./checkpoints/" + opt.exp_name,
                monitor="train_psnr",  # "train_loss",
                mode="max",
                save_top_k=3,
                save_last=True,
                filename="{epoch}-{train_psnr:.4f}",
            ),
        ],
    )

    # Start training
    trainer.fit(
        Diffusion(opt, diffusion, len(dataloader)),
        train_dataloaders=dataloader,
    )
    wandb.finish()
