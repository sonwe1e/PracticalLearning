import shutil
import os
from dataset import *
from pl_tool import *
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import torch
import yaml
import wandb


torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    """获取 config"""
    with open("./config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    """定义网络"""
    from model.vae import vae

    model = vae(config)
    "Compile"
    # model = torch.compile(model)

    # Define dataloader
    train_dataloader, valid_dataloader = get_dataloader(config)

    # Training Init
    pl.seed_everything(config["seed"])
    wandb_logger = WandbLogger(
        project=config["project"],
        name=config["exp_name"],
        offline=not config["save_wandb"],
        config=config,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config["epochs"],
        precision="bf16-mixed",
        default_root_dir="./",
        deterministic=False,
        logger=wandb_logger,
        val_check_interval=config["val_check"],
        log_every_n_steps=5,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=f"./checkpoints/" + config["exp_name"],
                monitor="valid_loss",
                mode="min",
                save_top_k=1,
                save_last=False,
                filename="{epoch}-{valid_loss:.4f}",
            ),
        ],
    )

    # Start training
    trainer.fit(
        VAE(config, model, len(train_dataloader)),
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
