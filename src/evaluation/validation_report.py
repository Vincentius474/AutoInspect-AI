import torch
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from src.data.loaders import create_dataloaders
from src.models.custom_cnn import CustomCNN
from src.models.mobilenet import create_mobilenet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Low", "Medium", "High"]

def evaluate_model(model, checkpoint_path, model_name):
    """Evaluate a model on the validation set and generate a classification report and confusion matrix."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    _, val_loader, _ = create_dataloaders(
        data_dir="data/splits",
        batch_size=16,
        image_size=224,
        num_workers=0
    )

    actual = []
    predicted = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            actual.extend(labels.tolist())
            predicted.extend(predictions.cpu().tolist())

    print(f"\n{'=' * 50}")
    print(model_name)
    print("=" * 50)

    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1, 2]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} Validation Confusion Matrix")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(f"reports/figures/{safe_name}_validation_cm.png", dpi=150)
    plt.close()

def main():
    """Main function to evaluate models and generate reports."""
    custom_model = CustomCNN(num_classes=3)

    evaluate_model(
        custom_model,
        "models/checkpoints/best_custom_cnn.pth",
        "Custom CNN"
    )

    mobilenet_model = create_mobilenet(
        num_classes=3,
        pretrained=False
    )

    evaluate_model(
        mobilenet_model,
        "models/checkpoints/best_mobilenet.pth",
        "MobileNetV3-Small"
    )

if __name__ == "__main__":
    main()
