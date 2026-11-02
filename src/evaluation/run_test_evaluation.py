from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)
from src.data.loaders import create_dataloaders
from src.models.mobilenet import create_mobilenet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = Path("models/checkpoints/best_mobilenet.pth")
REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"
CLASS_NAMES = ["Low", "Medium", "High"]

def main():
    """Run the final evaluation on the test dataset and save metrics and confusion matrix."""
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print("Device:", DEVICE)

    _, _, test_loader = create_dataloaders(
        data_dir="data/splits",
        batch_size=16,
        image_size=224,
        num_workers=0
    )

    model = create_mobilenet(
        num_classes=3,
        pretrained=False
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Loaded checkpoint from epoch:", checkpoint["epoch"])
    print("Validation macro F1:", checkpoint["best_val_f1"])

    actual = []
    predicted = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            actual.extend(labels.tolist())
            predicted.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(actual, predicted)
    balanced_accuracy = balanced_accuracy_score(
        actual,
        predicted
    )

    macro_precision = precision_score(
        actual,
        predicted,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        actual,
        predicted,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        actual,
        predicted,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        actual,
        predicted,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1, 2]
    )

    print("\nFINAL TEST RESULTS")
    print("=" * 50)

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Macro Precision:   {macro_precision:.4f}")
    print(f"Macro Recall:      {macro_recall:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")

    print("\nClassification Report")
    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("MobileNetV3-Small - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "final_test_confusion_matrix.png", dpi=150)
    plt.close()

    results = {
        "model": "MobileNetV3-Small",
        "test_samples": len(actual),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": matrix.tolist()
    }

    with open(REPORT_DIR / "final_test_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nSaved:")
    print("reports/final_test_results.json")
    print("reports/figures/final_test_confusion_matrix.png")

if __name__ == "__main__":
    main()
