import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ChestXrayDataset


image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

train_dataset = ChestXrayDataset(
    csv_file="data/train.csv",
    data_folder="data",
    transform=image_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

images, labels = next(iter(train_loader))

print("Number of images in dataset:", len(train_dataset))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Labels:", labels)
print("Image data type:", images.dtype)