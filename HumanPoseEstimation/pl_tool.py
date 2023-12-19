import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from lion_pytorch import Lion
import numpy as np

torch.set_float32_matmul_precision("high")


class HumanPoseEstimation(pl.LightningModule):
    def __init__(self, config, model, len_trainloader):
        super().__init__()
        self.learning_rate = config["learning_rate"]
        self.len_trainloader = len_trainloader
        self.config = config
        self.model = model
        self.criterion = torch.nn.MSELoss(reduction="mean")

    def forward(self, x):
        pred = self.model(x)
        return pred

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
        image, target_cl = train_batch
        pred_cl = self.model(image)
        loss = self.criterion(pred_cl[-1], target_cl)

        self.log("train_loss", loss)
        pckh50 = self.calculate_pckh_torch(pred_cl[-1], target_cl)
        self.log("train_pckh50", pckh50)
        self.log("trainer/learning_rate", self.scheduler.get_last_lr()[0])
        return loss

    def validation_step(self, val_batch, batch_idx):
        image, target_cl = val_batch
        pred_cl = self.model(image)
        loss = self.criterion(pred_cl[-1], target_cl)
        pckh50 = self.calculate_pckh_torch(pred_cl[-1], target_cl)
        self.log("valid_pckh50", pckh50)
        self.log("valid_loss", loss)

    def calculate_pckh_torch(self, prediction, ground_truth, head_segment_length=0.5):
        """
        Calculate PCKh@0.5 metric for human pose estimation using PyTorch.

        :param prediction: PyTorch tensor of predicted keypoints, shape (b, num_joints, h, w)
        :param ground_truth: PyTorch tensor of ground truth keypoints, shape (b, num_joints, h, w)
        :param head_segment_length: length of the head segment
        :return: PCKh@0.5 score
        """
        threshold = 0.5 * head_segment_length
        batch_size, num_joints, height, width = prediction.shape

        # Get the coordinates of the maximum values (keypoints) in the heatmaps
        pred_coords = torch.argmax(prediction.view(batch_size, num_joints, -1), dim=2)
        gt_coords = torch.argmax(ground_truth.view(batch_size, num_joints, -1), dim=2)

        pred_coords = torch.stack((pred_coords // width, pred_coords % width), dim=2)
        gt_coords = torch.stack((gt_coords // width, gt_coords % width), dim=2)

        # Calculate the Euclidean distance between predicted and ground truth keypoints
        distances = torch.norm(pred_coords.float() - gt_coords.float(), dim=2)

        # Calculate PCKh@0.5
        correct_keypoints = (distances <= threshold).float().sum()
        total_keypoints = torch.numel(distances)

        return correct_keypoints / total_keypoints

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass
