import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.data.preprocess import NUM_CLASSES

CHECKPOINT_METRIC = "macro_f1"


class EpochMetrics:
    def __init__(self):
        self.samples = 0
        self.total_loss = 0.0
        self.correct = 0
        self.confusion = torch.zeros(
            NUM_CLASSES,
            NUM_CLASSES,
            dtype=torch.long
        )

    def update(self, batch_size, loss, predicted_class, true_class):
        self.samples += batch_size
        self.total_loss += loss.item() * batch_size
        self.correct += (predicted_class == true_class).sum().item()

        confusion_indices = (
            true_class.detach().cpu() * NUM_CLASSES
            + predicted_class.detach().cpu()
        )
        self.confusion += torch.bincount(
            confusion_indices,
            minlength=NUM_CLASSES * NUM_CLASSES
        ).reshape(NUM_CLASSES, NUM_CLASSES)

    def compute(self):
        if self.samples == 0:
            raise RuntimeError("Cannot calculate metrics from an empty loader")

        true_positive = self.confusion.diag().float()
        actual = self.confusion.sum(dim=1).float()
        predicted = self.confusion.sum(dim=0).float()

        precision = true_positive / predicted.clamp_min(1)
        recall = true_positive / actual.clamp_min(1)
        f1 = (
            2 * precision * recall
            / (precision + recall).clamp_min(1e-8)
        )

        return {
            "loss": self.total_loss / self.samples,
            "accuracy": self.correct / self.samples,
            "balanced_accuracy": recall.mean().item(),
            "macro_precision": precision.mean().item(),
            "macro_recall": recall.mean().item(),
            "macro_f1": f1.mean().item(),
            "recall_per_class": recall.tolist(),
            "confusion_matrix": self.confusion.tolist()
        }


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    metrics = EpochMetrics()
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for batch in loader:
            images = batch["image"].to(device)
            true_class = batch["class_id"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, true_class)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            metrics.update(
                images.size(0), loss,
                outputs.argmax(dim=1), true_class
            )

    return metrics.compute()


def evaluate(model, loader, device):
    return run_epoch(model, loader, nn.CrossEntropyLoss(), device)


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad)


def scalar_metrics(metrics):
    """Bỏ recall từng lớp và confusion matrix để history gọn."""
    return {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, list)
    }


def save_json(data, output_path):
    Path(output_path).write_text(
        json.dumps(data, indent=2),
        encoding="utf-8")


def save_confusion_matrix(matrix, class_names, output_path):
    pd.DataFrame(
        matrix, index=class_names, columns=class_names
    ).to_csv(output_path, index_label="true \\ predicted", encoding="utf-8-sig")


def save_best_metrics(output_dir, model_name, epoch, val_metrics, class_names,
                      test_metrics=None):
    data = {
        "model": model_name,
        "best_epoch": epoch,
        "checkpoint_metric": CHECKPOINT_METRIC,
        "val": scalar_metrics(val_metrics),
        "val_recall_per_class": dict(
            zip(class_names, val_metrics["recall_per_class"])
        )
    }

    if test_metrics is not None:
        data["test"] = scalar_metrics(test_metrics)
        data["test_recall_per_class"] = dict(
            zip(class_names, test_metrics["recall_per_class"])
        )

    save_json(data, Path(output_dir) / "best_metrics.json")


def append_run_summary(summary_path, row):
    """Thêm một dòng vào bảng tổng hợp mọi lần chạy (không ghi đè)."""
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        summary_path,
        mode="a",
        header=not summary_path.exists(),
        index=False
    )


def save_checkpoint(output_path, model, optimizer, epoch, metrics, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": scalar_metrics(metrics),
        "config": config
    }, output_path)


def train_model(model, train_loader, val_loader, device, output_dir, model_name,
                max_epochs, learning_rate, weight_decay, patience,
                class_names, class_weights=None, extra_config=None
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )

    config = {
        "model": model_name,
        "parameter_count": count_parameters(model),
        "num_classes": NUM_CLASSES,
        "max_epochs": max_epochs,
        "optimizer": {
            "name": optimizer.__class__.__name__,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay
        },
        "scheduler": "ReduceLROnPlateau(mode=max, factor=0.5, patience=4)",
        "early_stopping_patience": patience,
        "loss": "CrossEntropyLoss",
        "class_weights": (
            class_weights.tolist() if class_weights is not None else None
        ),
        "checkpoint_metric": f"val_{CHECKPOINT_METRIC}",
        "device": str(device)
    }
    config.update(extra_config or {})
    save_json(config, output_dir / "config.json")

    checkpoint_path = output_dir / "best_model.pt"
    history = []
    best = None
    best_checkpoint_score = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        start_time = time.perf_counter()
        learning_rate_used = optimizer.param_groups[0]["lr"]

        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics[CHECKPOINT_METRIC])

        history.append({
            "epoch": epoch,
            "learning_rate": learning_rate_used,
            "seconds": time.perf_counter() - start_time,
            "train": scalar_metrics(train_metrics),
            "val": scalar_metrics(val_metrics)
        })
        save_json(history, output_dir / "history.json")

        checkpoint_score = val_metrics[CHECKPOINT_METRIC]
        if checkpoint_score > best_checkpoint_score:
            best_checkpoint_score = checkpoint_score
            epochs_without_improvement = 0
            best = {"epoch": epoch, "val": val_metrics}

            save_checkpoint(
                checkpoint_path, model, optimizer, epoch, val_metrics, config)
            save_best_metrics(
                output_dir, model_name, epoch, val_metrics, class_names)
            save_confusion_matrix(
                val_metrics["confusion_matrix"], class_names,
                output_dir / "confusion_matrix_val.csv")
        else:
            epochs_without_improvement += 1

        print(
            f"{model_name} epoch {epoch:02d}/{max_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"lr={learning_rate_used:.2e} | "
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    return history, checkpoint_path, best
