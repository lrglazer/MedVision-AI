from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from dataset import ChestXrayDataset


BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/best_model.pt")


def evaluate_model(model, loader, loss_function):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            logits = model(images)
            loss = loss_function(logits, labels)
            probabilities = torch.sigmoid(logits)

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probabilities.extend(
                probabilities.cpu().numpy().flatten()
            )

    average_loss = total_loss / len(loader)

    try:
        auc = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        auc = 0.0

    return average_loss, auc


def main():
    print("Using device:", DEVICE)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    validation_transform = transforms.Compose(
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
        transform=train_transform,
    )

    validation_dataset = ChestXrayDataset(
        csv_file="data/val.csv",
        data_folder="data",
        transform=validation_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze the pretrained image-processing layers.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace the final classification layer.
    feature_count = model.fc.in_features
    model.fc = nn.Linear(feature_count, 1)

    model = model.to(DEVICE)

    positive_count = int(
        (train_dataset.labels["label"] == 1).sum()
    )
    negative_count = int(
        (train_dataset.labels["label"] == 0).sum()
    )

    positive_weight = negative_count / positive_count

    print("Negative training images:", negative_count)
    print("Positive training images:", positive_count)
    print("Positive class weight:", round(positive_weight, 2))

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [positive_weight],
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_auc = 0.0

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        for batch_number, (images, labels) in enumerate(
            train_loader,
            start=1,
        ):
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = loss_function(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_number % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_number}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        training_loss = running_loss / len(train_loader)

        validation_loss, validation_auc = evaluate_model(
            model,
            validation_loader,
            loss_function,
        )

        print()
        print(f"Epoch {epoch + 1} complete")
        print(f"Training loss:   {training_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation AUC:  {validation_auc:.4f}")
        print()

        if validation_auc > best_validation_auc:
            best_validation_auc = validation_auc

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

            print(f"Saved improved model to {MODEL_PATH}")
            print()

    print("Training finished.")
    print(f"Best validation AUC: {best_validation_auc:.4f}")


if __name__ == "__main__":
    main()