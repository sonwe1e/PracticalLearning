import json
import torch
from option import get_option
import shutil
import os
from dataset import *
from pl_tool import *
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import torch
from torchvision import models
import wandb


torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    opt = get_option()
    """定义网络"""
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # model.fc = torch.nn.Linear(512, 200)
    from ImageClassification.model import inceptionv1

    model = inceptionv1.inceptionv1()

    """模型编译"""
    # model = torch.compile(model)

    """导入数据集"""
    train_dataloader, valid_dataloader = get_dataloader(opt)

    """Lightning 模块定义"""
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
        log_every_n_steps=opt.log_step,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=f"./ImageClassification/checkpoints/" + opt.exp_name,
                monitor="valid/valid_f1",
                mode="max",
                save_top_k=1,
                save_last=True,
                filename="{epoch}-{valid/valid_f1:.2f}",
            ),
        ],
    )

    # Start training
    trainer.fit(
        LightningModule(opt, model, len(train_dataloader)),
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    wandb.finish()
