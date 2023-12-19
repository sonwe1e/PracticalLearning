import numpy as np
import cv2
import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import json
import time

train_transform = A.Compose(
    [
        A.Normalize(),
        ToTensorV2(),
    ],
)


test_transform = A.Compose(
    [
        A.Normalize(),
        ToTensorV2(),
    ],
)


def get_dataloader(conf):
    train_dataset = MPIIDataset(
        conf["root_path"], train=True, transform=train_transform
    )
    test_dataset = MPIIDataset(conf["root_path"], train=False, transform=test_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=conf["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=conf["batch_size"],
        shuffle=False,
        num_workers=conf["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, test_loader


class MPIIDataset(Dataset):
    def __init__(
        self,
        root_path,
        image_size=(320, 640),
        train=True,
        transform=None,
    ):
        image_set = "train" if train else "valid"
        file_name = os.path.join(root_path, "annot", image_set + ".json")
        self.num_joints = 16
        self.transform = transform
        self.image_dict = {}
        self.label_dict = {}
        self.root_path = root_path
        self.image_size = image_size
        with open(file_name) as anno_file:
            self.annos = json.load(anno_file)
        self.process_images_in_parallel()

    def process_anno(self, anno, root_path, image_size):
        anno_index = anno["image"]
        image = cv2.imread(os.path.join(root_path, "images", anno_index))
        origin_size = image.shape[:2]
        self.image_dict[anno_index] = cv2.resize(image, image_size)
        joint = np.array(anno["joints"]).reshape(-1, 2)
        joint = joint * np.array(image_size) / np.flip(np.array(origin_size))
        if anno_index not in self.label_dict:
            self.label_dict[anno_index] = joint
        else:
            self.label_dict[anno_index] = np.vstack(
                (self.label_dict[anno_index], joint)
            )

    def process_images_in_parallel(self):
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.process_anno, anno, self.root_path, self.image_size
                )
                for anno in self.annos
            ]
            for future in futures:
                future.result()  # 等待每个任务完成

    def __getitem__(self, index):
        image_index = self.annos[index]["image"]
        image = self.image_dict[image_index]
        label = self.generate_heatmap(image, self.label_dict[image_index])

        if self.transform is not None:
            tranformed_data = self.transform(image=image, mask=label)
            image, label = tranformed_data["image"], tranformed_data["mask"]
        label = label.permute(2, 0, 1)
        label = (
            torch.nn.functional.interpolate(
                label.unsqueeze(0), size=(80, 40), mode="bilinear"
            )
            .squeeze(0)
            .float()
        )
        return image, label

    def __len__(self):
        return len(self.annos)

    def generate_heatmap(self, image, label, sigma=3):
        heatmap = np.zeros((*image.shape[:2], self.num_joints))
        for _, p in enumerate(label):
            if p[0] < 0 or p[1] < 0:
                continue
            if p[1] >= image.shape[0]:
                h = image.shape[0] - 1
            else:
                h = int(p[1])
            if p[0] >= image.shape[1]:
                w = image.shape[1] - 1
            else:
                w = int(p[0])
            heatmap[h, w, _ % self.num_joints] = 1
        heatmap = cv2.GaussianBlur(heatmap, (13, 13), sigma)

        return heatmap


if __name__ == "__main__":
    train_set = MPIIDataset(
        "/home/sonwe1e/WorkStation/Dataset/MPII/", transform=train_transform
    )
    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=False,
        persistent_workers=True,
    )
    for image, label in tqdm(train_loader):
        print(image.shape, label.shape)
