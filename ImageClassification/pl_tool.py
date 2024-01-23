import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from lion_pytorch import Lion
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

torch.set_float32_matmul_precision("high")


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader):
        super().__init__()
        self.learning_rate = opt.learning_rate
        self.len_trainloader = len_trainloader
        self.opt = opt
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_pred = []
        self.train_true = []
        self.valid_pred = []
        self.valid_true = []

    def forward(self, x):
        pred = self.model(x)
        return pred

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        self.train_pred.extend(pred.cpu().numpy())
        self.train_true.extend(y.cpu().numpy())

        self.log("train/train_loss", loss)
        self.log("learning_rate", self.scheduler.get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        self.valid_pred.extend(pred.cpu().numpy())
        self.valid_true.extend(y.cpu().numpy())

        self.log("valid/valid_loss", loss)

    def on_train_epoch_end(self):
        precision, recall, f1 = self.calculate_metrics(
            np.array(self.train_pred), np.array(self.train_true)
        )
        self.log("train/train_precision", precision)
        self.log("train/train_recall", recall)
        self.log("train/train_f1", f1)
        self.train_pred = []
        self.train_true = []

    def on_validation_epoch_end(self):
        precision, recall, f1 = self.calculate_metrics(
            np.array(self.valid_pred), np.array(self.valid_true)
        )
        self.log("valid/valid_precision", precision)
        self.log("valid/valid_recall", recall)
        self.log("valid/valid_f1", f1, prog_bar=True)
        self.valid_pred = []
        self.valid_true = []

    def calculate_metrics(self, pred, y):
        precision = precision_score(y, pred, average="macro")
        recall = recall_score(y, pred, average="macro")
        f1 = f1_score(y, pred, average="macro")
        return precision, recall, f1
