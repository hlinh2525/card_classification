from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import hashlib
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIP_PATH = PROJECT_ROOT / "data/raw/dataset.zip"
METADATA_DIR = PROJECT_ROOT / "data/metadata"
OUTPUT_DIR = PROJECT_ROOT / "data/preprocessing"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 53

SPLIT_NAMES = ("train", "val", "test")
COLUMNS = ["member", "class_name", "class_id"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_splits(metadata_dir=METADATA_DIR):
    metadata_dir = Path(metadata_dir)

    return {
        split: pd.read_csv(
            metadata_dir / f"{split}.csv",
            usecols=list(COLUMNS),
            dtype={
                "member": "string",
                "class_name": "string",
                "class_id": "int64"
            },
            encoding="utf-8-sig"
        )
        for split in SPLIT_NAMES
    }


def create_zip_index(zip_path, members):
    zip_path = Path(zip_path)

    if not zip_path.is_file():
        raise FileNotFoundError(f"Card ZIP not found: {zip_path}")

    requested_members = {
        str(member).strip().replace("\\", "/")
        for member in members
    }

    with ZipFile(zip_path) as archive:
        zip_members = set(archive.namelist())

    missing = requested_members - zip_members

    if missing:
        raise FileNotFoundError(
            f"ZIP is missing {len(missing)} images. Example: {sorted(missing)[:3]}"
        )

    return {member: member for member in requested_members}


def create_transform(model_type, training, image_size=IMAGE_SIZE):
    if model_type not in ["scratch", "resnet18"]:
        raise ValueError("model_type must be either scratch or resnet18.")

    steps = [
        transforms.Resize((image_size, image_size))
    ]

    if training:
        steps.extend([
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1
            )
        ])

    steps.append(transforms.ToTensor())

    if model_type == "resnet18":
        steps.append(
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        )

    return transforms.Compose(steps)


class CardDataset(Dataset):
    def __init__(self, data, zip_path, index, transform):
        self.data = data.reset_index(drop=True)
        self.zip_path = Path(zip_path)
        self.index = index
        self.transform = transform
        self.archive = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, position):
        row = self.data.iloc[position]

        if self.archive is None:
            self.archive = ZipFile(self.zip_path)

        content = self.archive.read(self.index[row["member"]])

        with Image.open(BytesIO(content)) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {
            "image": image,
            "class_id": torch.tensor(
                int(row["class_id"]),
                dtype=torch.long
            ),
            "member": row["member"]
        }

    def close(self):
        if self.archive is not None:
            self.archive.close()
            self.archive = None


def build_loaders(
    model_type="scratch",
    metadata_dir=METADATA_DIR,
    zip_path=ZIP_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
):
    if image_size < 32 or batch_size < 1:
        raise ValueError("image_size >= 32 and batch_size >= 1.")

    splits = load_splits(metadata_dir)

    members = pd.concat(splits.values())["member"]
    index = create_zip_index(zip_path, members)

    loaders = {}

    for name, data in splits.items():
        transform = create_transform(
            model_type=model_type,
            training=(name == "train"),
            image_size=image_size
        )

        dataset = CardDataset(
            data=data,
            zip_path=zip_path,
            index=index,
            transform=transform
        )

        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=False,
            num_workers=2
        )

    return loaders


def check_batches(loaders, model_type, image_size=IMAGE_SIZE):
    for name, loader in loaders.items():
        batch = next(iter(loader))
        size = len(batch["member"])

        assert batch["image"].shape == (size, 3, image_size, image_size)
        assert batch["class_id"].shape == (size,)

        assert batch["image"].dtype == torch.float32
        assert batch["class_id"].dtype == torch.int64

        assert torch.isfinite(batch["image"]).all()
        assert batch["class_id"].ge(0).all()
        assert batch["class_id"].lt(NUM_CLASSES).all()

        if model_type == "scratch":
            assert batch["image"].min() >= 0
            assert batch["image"].max() <= 1

        print(model_type, name, tuple(batch["image"].shape))

    for name in ["val", "test"]:
        dataset = loaders[name].dataset

        assert torch.equal(
            dataset[0]["image"],
            dataset[0]["image"]
        )


def save_augmentation_preview(loader, model_type, output_path):
    dataset = loader.dataset

    figure, axes = plt.subplots(1, 4, figsize=(12, 3))

    for axis in axes:
        image = dataset[0]["image"].permute(1, 2, 0)

        if model_type == "resnet18":
            image = (
                image * torch.tensor(IMAGENET_STD)
                + torch.tensor(IMAGENET_MEAN)
            )

        axis.imshow(image.clamp(0, 1).numpy())
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_config(output_dir):
    label_map = json.loads(
        (METADATA_DIR / "label_map.json").read_text(encoding="utf-8")
    )

    assert len(label_map) == NUM_CLASSES, (
        f"label_map has {len(label_map)} classes, expected {NUM_CLASSES}."
    )

    config = {
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "color": "RGB",
        "batch_size": BATCH_SIZE,
        "target": "class_id",
        "num_classes": NUM_CLASSES,
        "label_map": label_map,
        "scratch": {"pixel_range": [0, 1]},
        "resnet18": {
            "pixel_range_before_normalize": [0, 1],
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD
        },
        "train_augmentation": {
            "rotation_degrees": 10,
            "brightness": 0.1,
            "contrast": 0.1,
            "horizontal_flip": False
        },
        "validation_test_augmentation": False,
        "split_sha256": {
            name: hashlib.sha256(
                (METADATA_DIR / f"{name}.csv").read_bytes()
            ).hexdigest()
            for name in SPLIT_NAMES
        }
    }

    path = output_dir / "preprocessing_config.json"
    path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_config(OUTPUT_DIR)

    for model_type in ["scratch", "resnet18"]:
        loaders = build_loaders(model_type=model_type)

        try:
            check_batches(loaders, model_type)

            save_augmentation_preview(
                loaders["train"],
                model_type,
                OUTPUT_DIR / f"augmentation_{model_type}.png"
            )
        finally:
            for loader in loaders.values():
                loader.dataset.close()

    print("Preprocessing completed.")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
