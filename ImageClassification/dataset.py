import os
import torch
from torchvision import transforms
from torchvision import datasets

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ]
)


def get_dataloader(opt):
    train_dataset = datasets.ImageFolder(
        root=os.path.join(opt.dataset_root, "train"), transform=train_transform
    )
    valid_dataset = datasets.ImageFolder(
        root=os.path.join(opt.dataset_root, "val"), transform=test_transform
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    return train_dataloader, valid_dataloader
