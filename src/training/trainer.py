import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.models.custom_cnn import CustomCNN
from src.data.loaders import create_dataloaders
from src.training.metrics import calculate_metrics

def set_seed(seed=42):
    """Sets the random seed for reproducibility across various libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def run_epoch(model, loader, criterion, optimizer, device, training):
    """Runs a single epoch of training or validation."""
    model.train(training)
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:

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

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(labels.cpu().numpy())

    average_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )
    return average_loss, metrics

def train_model(epochs=40, batch_size=16, learning_rate=0.001, patience=8, seed=42):
    """Trains the CustomCNN model on the training dataset and evaluates it on the validation dataset."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, _ = create_dataloaders(
        data_dir="data/splits",
        batch_size=batch_size,
        image_size=224,
        num_workers=0
    )

    model = CustomCNN(num_classes=3).to(device)
    labels = [
        label
        for _, label in train_loader.dataset.samples
    ]
    counts = np.bincount(labels, minlength=3)

    if np.any(counts == 0):
        raise ValueError("A training class contains no images.")

    weights = len(labels) / (3.0 * counts)
    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=5e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": []
    }

    for epoch in range(epochs):

        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True
        )

        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            training=False
        )

        scheduler.step(val_metrics["f1"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch [{epoch + 1}/{epochs}] "
            f"LR: {current_lr:.6f}"
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1 + 1e-4:

            best_f1 = val_metrics["f1"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_f1": best_f1,
                    "class_to_idx": {
                        "low": 0,
                        "medium": 1,
                        "high": 2
                    }
                },
                checkpoint_dir / "best_custom_cnn.pth"
            )

            print("Saved new best model!")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    return history

if __name__ == "__main__":
    from src.training.plot_history import plot_history
    history = train_model(
        epochs=50,
        batch_size=16,
        learning_rate=0.0005,
        patience=10,
        seed=42
    )
    plot_history(history)
