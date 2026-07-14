from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from dataset import ChestXrayDataset


BATCH_SIZE = 32
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/best_model.pt")


def main():
    print("Using device:", DEVICE)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Run train.py first."
        )

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

    test_dataset = ChestXrayDataset(
        csv_file="data/test.csv",
        data_folder="data",
        transform=image_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = resnet18(weights=None)

    feature_count = model.fc.in_features
    model.fc = nn.Linear(feature_count, 1)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model = model.to(DEVICE)
    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            logits = model(images)
            probabilities = torch.sigmoid(logits).squeeze(1)

            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    predictions = [
        1 if probability >= THRESHOLD else 0
        for probability in all_probabilities
    ]

    auc = roc_auc_score(all_labels, all_probabilities)

    matrix = confusion_matrix(
        all_labels,
        predictions,
        labels=[0, 1],
    )

    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )

    sensitivity = true_positive / max(
        true_positive + false_negative,
        1,
    )

    specificity = true_negative / max(
        true_negative + false_positive,
        1,
    )

    print("\nTest results")
    print(f"ROC-AUC:     {auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    print("\nConfusion matrix")
    print(matrix)

    print("\nClassification report")
    print(
        classification_report(
            all_labels,
            predictions,
            target_names=["No Finding", "Pneumonia"],
            digits=4,
            zero_division=0,
        )
    )

    false_positive_rate, true_positive_rate, _ = roc_curve(
        all_labels,
        all_probabilities,
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {auc:.3f}",
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Chest X-Ray Model ROC Curve")
    plt.legend()
    plt.tight_layout()

    output_path = Path("models/roc_curve.png")
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"\nSaved ROC curve to {output_path}")


if __name__ == "__main__":
    main()