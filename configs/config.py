from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_PATH = RAW_DATA_DIR / "dataset.zip"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
IMAGE_SIZE = (224, 224)
EXPECTED_CLASSES = 53
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
