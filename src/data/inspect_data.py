import io
import json
import re
import zipfile
from hashlib import sha256
from pathlib import Path, PurePosixPath
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ARCHIVE = PROJECT_ROOT / "data" / "raw" / "dataset.zip"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

IMAGE_SIZE = (224, 224)
EXPECTED_CLASSES = 53
RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

METADATA_COLUMNS = ["member", "class_name", "class_id", "split"]

METADATA_DIR.mkdir(parents=True, exist_ok=True)


def get_class_name(member):
    """'dataset/dataset/ace of clubs 01.jpg' -> 'ace of clubs'."""
    match = re.match(r"^(.+?)\s+\d+$", PurePosixPath(member).stem)
    if match is None:
        return None
    return " ".join(match.group(1).lower().split())


def inspect_dataset():
    if not RAW_ARCHIVE.is_file():
        raise FileNotFoundError(f"Dataset archive was not found: {RAW_ARCHIVE}")

    records = []
    skipped = []

    with zipfile.ZipFile(RAW_ARCHIVE) as archive:
        for member in sorted(archive.namelist()):
            parts = PurePosixPath(member).parts

            if member.endswith("/") or any(
                p.startswith(".") or p == "__MACOSX" for p in parts
            ):
                continue

            class_name = get_class_name(member)
            if class_name is None:
                skipped.append((member, "filename is not '<class name> <number>'"))
                continue

            try:
                image_bytes = archive.read(member)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.load()
                    width, height = image.size
                    channels = len(image.getbands())
            except (OSError, ValueError, Image.DecompressionBombError):
                skipped.append((member, "not a readable image"))
                continue

            records.append(
                {
                    "member": member,
                    "class_name": class_name,
                    "sha256": sha256(image_bytes).hexdigest(),
                    "width": width,
                    "height": height,
                    "channels": channels,
                }
            )

    metadata = pd.DataFrame(records)

    print("\n" + "=" * 60)
    print("CARD DATASET INSPECTION")
    print("=" * 60)
    print(f"Valid images      : {len(metadata):,}")
    print(f"Skipped files     : {len(skipped):,}")
    for member, reason in skipped[:10]:
        print(f"  - {member}: {reason}")

    if not metadata.empty:
        print(f"Image dimensions  : {metadata[['width', 'height']].value_counts().to_dict()}")
        print(f"Channels          : {metadata['channels'].value_counts().sort_index().to_dict()}")
        print(f"Number of classes : {metadata['class_name'].nunique():,}")

        print("\nImages per class:")
        print(metadata["class_name"].value_counts().sort_index().to_string())

    return metadata


def clean_metadata(metadata):
    before = len(metadata)

    labels_per_image = metadata.groupby("sha256")["class_name"].transform("nunique")
    conflicts = metadata[labels_per_image > 1]
    clean = metadata[labels_per_image == 1]

    clean = clean.drop_duplicates("sha256").copy()
    duplicates_removed = before - len(conflicts) - len(clean)

    class_names = sorted(clean["class_name"].unique())
    class_to_id = {name: i for i, name in enumerate(class_names)}
    clean["class_id"] = clean["class_name"].map(class_to_id).astype("int64")

    with open(METADATA_DIR / "label_map.json", "w", encoding="utf-8") as file:
        json.dump(class_to_id, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)
    print(f"Before cleaning   : {before:,}")
    print(f"After cleaning    : {len(clean):,}")
    print(f"Duplicate removed : {duplicates_removed:,}")
    print(f"Conflict files    : {len(conflicts):,}")
    print(f"Number of classes : {len(class_names):,} (expected {EXPECTED_CLASSES})")

    if len(class_names) != EXPECTED_CLASSES:
        print(f"WARNING: expected {EXPECTED_CLASSES} classes.")

    return clean


def create_splits(metadata):
    if metadata["class_id"].value_counts().min() < 7:
        raise ValueError("Every class needs at least 7 images to split 70/15/15.")

    train, temp = train_test_split(
        metadata,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=metadata["class_id"],
    )
    val, test = train_test_split(
        temp,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_STATE,
        stratify=temp["class_id"],
    )

    splits = {
        "train": train.assign(split="train"),
        "val": val.assign(split="val"),
        "test": test.assign(split="test"),
    }

    print("\n" + "=" * 60)
    print("DATA SPLIT")
    print("=" * 60)
    for name, data in splits.items():
        data = data[METADATA_COLUMNS].sort_values(["class_id", "member"])
        data.to_csv(METADATA_DIR / f"{name}.csv", index=False)
        print(f"{name:<5}: {len(data):,} ({len(data) / len(metadata):.1%})")

    combined = pd.concat(splits.values())[METADATA_COLUMNS]
    combined = combined.sort_values(["class_id", "member"])
    combined.to_csv(METADATA_DIR / "clean_metadata.csv", index=False)

    print("\nImages per class by split:")
    print(pd.crosstab(combined["class_name"], combined["split"])[["train", "val", "test"]].to_string())


def main():
    print("=" * 60)
    print("CARD CLASSIFICATION DATA PREPARATION")
    print("=" * 60)
    print(f"Dataset path: {RAW_ARCHIVE}")
    print(f"Target size : {IMAGE_SIZE}")

    metadata = inspect_dataset()
    if metadata.empty:
        raise RuntimeError("No valid card images were found.")

    create_splits(clean_metadata(metadata))
    print(f"\nSaved metadata to: {METADATA_DIR}")


if __name__ == "__main__":
    main()
