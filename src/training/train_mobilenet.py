from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.models.mobilenet import create_mobilenet
from src.data.loaders import create_dataloaders


CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(labels, predictions):
    return {
        "accuracy": float(
            np.mean(np.array(labels) == np.array(predictions))
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0
            )
        )
    }


def run_epoch(model, loader, criterion, device, optimizer=None):

    training = optimizer is not None
    model.train(training)
    if training and not any(
        parameter.requires_grad
        for parameter in model.features.parameters()
    ):
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        progress = tqdm(loader, leave=False)

        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

            progress.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_predictions)

    return average_loss, metrics


def get_class_weights(dataset, device):
    labels = [label for _, label in dataset.samples]

    counts = np.bincount(labels, minlength=3)

    if np.any(counts == 0):
        raise ValueError("A class has no training images.")

    weights = len(labels) / (3.0 * counts)

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )


def save_checkpoint(model, epoch, best_f1, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_val_f1": best_f1,
            "class_to_idx": {
                "low": 0,
                "medium": 1,
                "high": 2
            }
        },
        path
    )


def train_mobilenet(
    warmup_epochs=5,
    fine_tune_epochs=25,
    batch_size=16,
    seed=42
):
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    train_loader, val_loader, _ = create_dataloaders(
        data_dir="data/splits",
        batch_size=batch_size,
        image_size=224,
        num_workers=0
    )

    model = create_mobilenet(
        num_classes=3,
        pretrained=True
    ).to(device)

    class_weights = get_class_weights(
        train_loader.dataset,
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    checkpoint_path = (
        CHECKPOINT_DIR / "best_mobilenet.pth"
    )

    best_val_f1 = -1.0
    best_epoch = 0

    # ---------------------------------------
    # Stage 1: Train classification layer
    # ---------------------------------------

    print("\nSTAGE 1: Training classifier")

    optimizer = AdamW(
        model.classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    for epoch in range(warmup_epochs):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer
        )

        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Warmup {epoch + 1}/{warmup_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1

            save_checkpoint(
                model,
                best_epoch,
                best_val_f1,
                checkpoint_path
            )

    # ---------------------------------------
    # Stage 2: Fine-tune entire network
    # ---------------------------------------

    print("\nSTAGE 2: Fine-tuning entire network")

    for parameter in model.features.parameters():
        parameter.requires_grad = True

    optimizer = AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": 1e-4
            },
            {
                "params": model.classifier.parameters(),
                "lr": 3e-4
            }
        ],
        weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    patience = 7
    epochs_without_improvement = 0

    for epoch in range(fine_tune_epochs):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer
        )

        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        val_f1 = val_metrics["macro_f1"]
        scheduler.step(val_f1)

        print(
            f"Fine-tune {epoch + 1}/{fine_tune_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train F1: {train_metrics['macro_f1']:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_epoch = warmup_epochs + epoch + 1
            epochs_without_improvement = 0

            save_checkpoint(
                model,
                best_epoch,
                best_val_f1,
                checkpoint_path
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    print("\nTraining complete.")
    print("Best epoch:", best_epoch)
    print(f"Best validation macro F1: {best_val_f1:.4f}")
    print("Checkpoint:", checkpoint_path)

    return best_val_f1

if __name__ == "__main__":
    train_mobilenet()
