import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import METADATA_DIR, NUM_CLASSES, build_loaders
from src.models.simple_cnn import SimpleCNN
from src.training.training import (
    append_run_summary,
    evaluate,
    save_best_metrics,
    save_confusion_matrix,
    train_model,
)


# Mỗi lần chạy lưu vào một thư mục riêng: data/results/simple_cnn/<thời gian>/
RESULTS_DIR = PROJECT_ROOT / "data/results"
MODEL_NAME = "simple_cnn"

MAX_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
SEED = 42


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_class_names():
    label_map = json.loads(
        (METADATA_DIR / "label_map.json").read_text(encoding="utf-8")
    )

    if len(label_map) != NUM_CLASSES:
        raise ValueError(
            f"label_map has {len(label_map)} classes, expected {NUM_CLASSES}."
        )
    if any(type(class_id) is not int for class_id in label_map.values()) or sorted(
        label_map.values()
    ) != list(range(NUM_CLASSES)):
        raise ValueError(f"Class IDs must be unique integers from 0 to {NUM_CLASSES - 1}.")

    return [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = RESULTS_DIR / MODEL_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    class_names = load_class_names()
    loaders = build_loaders(model_type="scratch")

    print(f"Run     : {MODEL_NAME}/{run_id}")
    print(f"Device  : {device}")
    print(f"Train   : {len(loaders['train'].dataset):,} images")
    print(f"Val     : {len(loaders['val'].dataset):,} images")
    print(f"Test    : {len(loaders['test'].dataset):,} images")

    start_time = time.perf_counter()

    try:
        model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
        history, checkpoint_path, best = train_model(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            device=device,
            output_dir=run_dir,
            model_name=MODEL_NAME,
            max_epochs=MAX_EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            class_names=class_names,
            extra_config={"run_id": run_id, "seed": SEED}
        )

        # Đánh giá trên tập test bằng model tốt nhất theo validation.
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, loaders["test"], device)
    finally:
        for loader in loaders.values():
            loader.dataset.close()

    save_best_metrics(
        run_dir, MODEL_NAME, best["epoch"], best["val"], class_names,
        test_metrics=test_metrics
    )
    save_confusion_matrix(
        test_metrics["confusion_matrix"], class_names,
        run_dir / "confusion_matrix_test.csv"
    )

    minutes = (time.perf_counter() - start_time) / 60
    val = best["val"]

    append_run_summary(RESULTS_DIR / "runs_summary.csv", {
        "run_id": run_id,
        "model": MODEL_NAME,
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "val_loss": val["loss"],
        "val_accuracy": val["accuracy"],
        "val_macro_f1": val["macro_f1"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "minutes": round(minutes, 2),
    })

    print("\n" + "=" * 60)
    print(f"{MODEL_NAME} RESULT")
    print("=" * 60)
    print(f"Best epoch        : {best['epoch']}")
    print(
        f"Best val          : loss={val['loss']:.4f} | "
        f"acc={val['accuracy']:.4f} | "
        f"balanced_acc={val['balanced_accuracy']:.4f} | "
        f"macro_f1={val['macro_f1']:.4f}"
    )
    print(
        f"Test              : loss={test_metrics['loss']:.4f} | "
        f"acc={test_metrics['accuracy']:.4f} | "
        f"balanced_acc={test_metrics['balanced_accuracy']:.4f} | "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )
    print(f"Training time     : {minutes:.1f} minutes")
    print(f"Saved results to  : {run_dir}")


if __name__ == "__main__":
    main()
