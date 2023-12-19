import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from Unet import Unet
from pathlib import Path

scaler = torch.cuda.amp.GradScaler()


class Dataset(Dataset):
    def __init__(
        self,
        data_path,
        image_size=32,
        file_types=["jpg", "jpeg", "png", "tiff"],
        H_flip=True,
        move2memory=False,
    ):
        super().__init__()
        self.data_path = data_path
        self.image_size = image_size
        self.paths = [
            p
            for file_type in file_types
            for p in Path(f"{data_path}").glob(f"**/*.{file_type}")
        ]
        self.move2memory = move2memory
        if move2memory:
            self.paths = [Image.open(p) for p in tqdm(self.paths)]
        # self.paths = self.paths[:100]

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip() if H_flip else nn.Identity(),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = self.paths[index] if self.move2memory else Image.open(self.paths[index])
        try:
            img = self.transform(img)
        except:
            print(img)
        return img


class GaussianDiffusion(torch.nn.Module):
    def __init__(
        self,
        model,
        beta1=1e-4,
        betaT=1e-2,
        T=1000,
        loss_type="l2",
        device=torch.device("cuda:0"),
    ):
        super().__init__()
        self.model = model.to(device)  # 将模型设置为设备
        self.loss_type = loss_type  # 设置损失类型
        self.T = T  # 初始化扩散系数的起始和结束值，以及时间步长
        self.betas = torch.linspace(beta1, betaT, T, device=device)  # 生成一个线性间隔的扩散系数向量
        self.alphas = 1 - self.betas  # 计算每个时间步的α值
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)  # 计算累积乘积的α_bar值
        self.alphas_bar_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_bar[:-1]]
        )  # 计算前一时间步的α_bar值

    def get_loss(self, x_0, t, loss_type="l2"):
        x_t, noise = self.get_noise_image(x_0, t)  # 获取加噪声的图像和噪声
        with torch.cuda.amp.autocast():
            predicted_noise = self.model(x_t, t)  # 使用模型预测噪声
            if self.loss_type == "l2":  # 如果损失类型为l2
                loss = F.mse_loss(predicted_noise, noise)  # 计算均方误差损失
            elif self.loss_type == "l1":  # 如果损失类型为l1
                loss = F.l1_loss(predicted_noise, noise)  # 计算L1损失
            else:
                raise ValueError("Unknown loss type")  # 如果损失类型未知，则抛出错误
        return loss

    def get_noise_image(self, x_0, t):
        sqrt_alphas_hat = torch.sqrt(self.alphas_bar[t])  # 计算sqrt(α_bar)
        sqrt_one_minus_alphas_hat = torch.sqrt(
            1 - self.alphas_bar[t]
        )  # 计算sqrt(1-α_bar)
        noise = torch.randn_like(x_0)  # 生成与x_0形状相同的随机噪声
        return (
            sqrt_alphas_hat[:, None, None, None] * x_0
            + sqrt_one_minus_alphas_hat[:, None, None, None] * noise,
            noise,
        )

    @torch.no_grad()
    def sample(self, n, T, image_size=(3, 32, 32), save_interval=True):
        self.model.eval()  # 将模型设置为评估模式
        samples = torch.randn(n, *image_size, device=self.betas.device)  # 生成初始随机样本
        if save_interval:
            interval = []
        for i in tqdm(reversed(range(1, T)), position=0):  # 从T逆序迭代到1
            t = (
                torch.ones(9, dtype=torch.long, device=self.betas.device) * i
            )  # 创建一个全是时间步i的张量
            predicted_noise = self.model(samples, t)  # 使用模型预测噪声
            alpha = self.alphas[t][:, None, None, None]  # 获取alpha值
            alpha_bar = self.alphas_bar[t][:, None, None, None]  # 获取alpha_bar值
            beta = self.betas[t][:, None, None, None]  # 获取beta值

            if i > 1:  # 如果不是最后一个时间步
                noise = torch.randn_like(samples)  # 生成噪声
            else:
                noise = torch.zeros_like(samples)  # 最后一个时间步不加噪声

            samples = (
                1
                / torch.sqrt(alpha)
                * (
                    samples
                    - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise
                )
                + torch.sqrt(beta) * noise
            )
            interval.append(samples) if save_interval else 0
        return samples, interval  # 返回生成的图像样本


class Trainer(object):
    def __init__(
        self,
        diffusion,
        data_path,
        device=torch.device("cuda:0"),
        image_size=128,
        batch_size=64,
        augment_horizontal_flip=True,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.diffusion = diffusion
        self.data_path = data_path
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), lr=learning_rate
        )
        self.device = device
        self.image_size = image_size

    def train(self, n_epochs):
        dataloader = self.prepare_data(self.data_path, self.batch_size)
        self.diffusion.train()
        best_loss = 1e10
        losses = []
        for epoch in range(n_epochs):
            for x in tqdm(dataloader):
                self.optimizer.zero_grad()
                x = x.to(self.device)
                t = torch.randint(
                    0, self.diffusion.T, (x.shape[0],), device=self.device
                )
                loss = self.diffusion.get_loss(x, t)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                losses.append(loss.item())

            # Print out the average loss for this epoch:
            avg_loss = sum(losses[-len(dataloader) :]) / len(dataloader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.diffusion.model.state_dict(), "best_model.pt")
            print(
                f"Finished epoch {epoch}. Average loss for this epoch: {avg_loss:.5f}"
            )

    def prepare_data(self, data_path, batch_size):
        dataset = Dataset(data_path, self.image_size)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
        )
        return dataloader


if __name__ == "__main__":
    Unet = Unet(3)
    diffusion = GaussianDiffusion(Unet, 1e-4, 1e-2, 1000)
    data_path = "/home/sonwe1e/WorkStation/Dataset/FFHQ/64x64/"
    trainer = Trainer(diffusion, data_path, batch_size=64, image_size=64)
    trainer.train(100)
