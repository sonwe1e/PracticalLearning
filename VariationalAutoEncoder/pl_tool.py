import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from lion_pytorch import Lion
import numpy as np

torch.set_float32_matmul_precision("high")


class VAE(pl.LightningModule):
    def __init__(self, config, model, len_trainloader):
        super().__init__()
        self.learning_rate = config["learning_rate"]
        self.len_trainloader = len_trainloader
        self.config = config
        self.model = model

    def forward(self, x):
        r_img, z, mu, log_var = self.model(x)
        return r_img

    def configure_optimizers(self):
        self.optimizer = Lion(
            self.parameters(),
            weight_decay=self.config["weight_decay"],
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.config["epochs"],
            pct_start=0.06,
            steps_per_epoch=self.len_trainloader,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def training_step(self, train_batch, batch_idx):
        image = train_batch
        # r_img, z, mu, std = self.model(image)
        # loss, recons_loss, kld_loss = self.model.loss_function(r_img, image, mu, std)
        r_image, z, loss = self.model(image)
        recons_loss = self.model.loss_function(r_image, image)
        self.log("train_loss", loss)
        self.log("train_recons_loss", recons_loss)
        # self.log("train_kld_loss", kld_loss)
        self.log("learning_rate", self.scheduler.get_last_lr()[0])
        return loss + recons_loss

    def validation_step(self, val_batch, batch_idx):
        image = val_batch
        # r_img, z, mu, std = self.model(image)
        # loss, recons_loss, kld_loss = self.model.loss_function(r_img, image, mu, std)
        r_image, z, loss = self.model(image)
        recons_loss = self.model.loss_function(r_image, image)
        self.log("valid_loss", loss)
        self.log("valid_recons_loss", recons_loss)
        # self.log("valid_kld_loss", kld_loss)
