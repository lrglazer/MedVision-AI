from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    def __init__(self, csv_file: str, data_folder: str, transform=None):
        self.data_folder = Path(data_folder)
        self.transform = transform

        csv_path = Path(csv_file)

        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

        if not self.data_folder.exists():
            raise FileNotFoundError(
                f"Could not find data folder: {self.data_folder}"
            )

        # Load the prepared train, validation, or test CSV.
        self.labels = pd.read_csv(csv_path)

        required_columns = {"Image Index", "Finding Labels", "label"}
        missing_columns = required_columns - set(self.labels.columns)

        if missing_columns:
            raise KeyError(
                f"CSV is missing required columns: {missing_columns}. "
                "Run prepare_data.py again."
            )

        # Find every PNG image inside the data folder and all subfolders.
        image_files = self.data_folder.rglob("*.png")

        self.image_paths = {
            image_path.name: image_path
            for image_path in image_files
        }

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No PNG images were found inside {self.data_folder}."
            )

        # Keep only rows whose image file actually exists.
        self.labels = self.labels[
            self.labels["Image Index"].isin(self.image_paths)
        ].copy()

        self.labels = self.labels.reset_index(drop=True)

        if len(self.labels) == 0:
            raise RuntimeError(
                "No CSV rows matched the image files. "
                "Check that the X-ray folders are inside the data folder."
            )

        # Ensure labels are numeric integers.
        self.labels["label"] = self.labels["label"].astype(int)

        print(f"Loaded {len(self.labels)} labeled X-rays.")

        label_counts = self.labels["label"].value_counts().sort_index()

        print("Class counts:")
        print(f"No Finding: {label_counts.get(0, 0)}")
        print(f"Pneumonia:  {label_counts.get(1, 0)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.labels):
            raise IndexError(
                f"Index {index} is outside dataset range."
            )

        row = self.labels.iloc[index]

        image_name = row["Image Index"]
        image_path = self.image_paths[image_name]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            raise RuntimeError(
                f"Could not open image: {image_path}"
            ) from error

        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label