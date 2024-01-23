import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from lion_pytorch import Lion
import numpy as np
import wandb
import copy

torch.set_float32_matmul_precision("high")


class Diffusion(pl.LightningModule):
    def __init__(self, opt, diffusion, len_trainloader):
        super().__init__()
        self.learning_rate = opt.learning_rate
        self.opt = opt
        self.len_trainloader = len_trainloader
        self.diffusion = diffusion
        # self.diffusion_ema = copy.deepcopy(diffusion)
        if opt.loss_type == "l1":
            self.criterion = F.l1_loss
        elif opt.loss_type == "l2":
            self.criterion = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, x):
        pass

    def configure_optimizers(self):
        self.optimizer = Lion(
            self.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.opt.epochs,
            pct_start=self.opt.pct_start,
            steps_per_epoch=self.len_trainloader,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        image = batch
        t = torch.randint(0, self.diffusion.T, (image.shape[0],), device=self.device)
        x_t, noise = self.diffusion.get_noise_image(image, t)
        # print(x_t.shape, noise.shape)
        predicted_noise = self.diffusion.model(x_t, t)
        loss = self.criterion(predicted_noise, noise)
        self.log("train_loss", loss)
        self.log("train_psnr", self.psnr(predicted_noise, noise))
        self.log("lr", self.scheduler.get_last_lr()[0])
        if batch_idx % (self.len_trainloader // self.opt.save_interval) == 0:
            generated_image = self.diffusion.ddim_sample(
                1, self.diffusion.T, 100, (3, self.opt.image_size, self.opt.image_size)
            )
            self.logger.experiment.log(
                {"generated_image": [wandb.Image(generated_image[0])]}
            )
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * torch.log10(255.0 / torch.sqrt(mse))
