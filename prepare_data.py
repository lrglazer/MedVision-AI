from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_FOLDER = Path("data")
SOURCE_CSV = DATA_FOLDER / "Data_Entry_2017.csv"


def main():
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Could not find: {SOURCE_CSV}")

    df = pd.read_csv(SOURCE_CSV)

    required_columns = {"Finding Labels", "Patient ID", "Image Index"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # Positive: Pneumonia appears anywhere in the multi-label string.
    pneumonia_mask = df["Finding Labels"].str.contains(
        "Pneumonia",
        regex=False,
        na=False,
    )

    # Negative: explicitly labeled No Finding.
    no_finding_mask = df["Finding Labels"].eq("No Finding")

    df = df[pneumonia_mask | no_finding_mask].copy()

    df["label"] = pneumonia_mask.loc[df.index].astype(int)

    print("Usable images:")
    print(df["label"].value_counts().rename(index={0: "No Finding", 1: "Pneumonia"}))

    patient_ids = df["Patient ID"].unique()

    train_patients, remaining_patients = train_test_split(
        patient_ids,
        test_size=0.30,
        random_state=42,
    )

    val_patients, test_patients = train_test_split(
        remaining_patients,
        test_size=0.50,
        random_state=42,
    )

    train_df = df[df["Patient ID"].isin(train_patients)].copy()
    val_df = df[df["Patient ID"].isin(val_patients)].copy()
    test_df = df[df["Patient ID"].isin(test_patients)].copy()

    train_df.to_csv(DATA_FOLDER / "train.csv", index=False)
    val_df.to_csv(DATA_FOLDER / "val.csv", index=False)
    test_df.to_csv(DATA_FOLDER / "test.csv", index=False)

    assert set(train_patients).isdisjoint(val_patients)
    assert set(train_patients).isdisjoint(test_patients)
    assert set(val_patients).isdisjoint(test_patients)

    for name, split_df in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Testing", test_df),
    ]:
        print(f"\n{name} images: {len(split_df)}")
        print(
            split_df["label"]
            .value_counts()
            .rename(index={0: "No Finding", 1: "Pneumonia"})
        )

    print("\nNo patient overlap found.")


if __name__ == "__main__":
    main()