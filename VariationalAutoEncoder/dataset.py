import numpy as np
from PIL import Image
import os
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


def train_transform(image, image_size=224):
    transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    return transform(image=image)["image"]


def test_transform(image, image_size=224):
    transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    return transform(image=image)["image"]


class VAE_dataset(Dataset):
    def __init__(
        self,
        image_path,
        train=True,
        transform=None,
        image_size=224,
    ):
        self.transform = transform
        self.image_size = image_size
        self.image_list = glob.glob(os.path.join(image_path, "*"))
        random.shuffle(self.image_list)
        self.image_list = (
            self.image_list[: int(len(self.image_list) * 0.8)]
            if train
            else self.image_list[int(len(self.image_list) * 0.8) :]
        )
        self.image_list = [
            np.asarray(Image.open(image)) for image in tqdm(self.image_list)
        ]

    def __getitem__(self, index):
        image = self.image_list[index]
        image = self.transform(image, self.image_size)
        return image

    def __len__(self):
        return len(self.image_list)


def get_dataloader(conf):
    train_dataset = VAE_dataset(
        image_path=conf["image_path"],
        train=True,
        transform=train_transform,
        image_size=conf["image_size"],
    )
    test_dataset = VAE_dataset(
        image_path=conf["image_path"],
        train=False,
        transform=test_transform,
        image_size=conf["image_size"],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=conf["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=conf["batch_size"],
        shuffle=False,
        num_workers=conf["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    print("Dataset Test")
