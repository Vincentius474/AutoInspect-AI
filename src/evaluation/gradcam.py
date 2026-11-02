import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import transforms
from src.models.mobilenet import create_mobilenet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Low", "Medium", "High"]
CHECKPOINT = "models/checkpoints/best_mobilenet.pth"
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def draw_gradcam_contours(image, grayscale_cam):
    """Draw contours on the original image based on the Grad-CAM heatmap."""
    base_image = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.uint8).copy()
    cam = cv2.resize(
        np.asarray(grayscale_cam, dtype=np.float32),
        (base_image.shape[1], base_image.shape[0]),
    )

    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=2)
    peak = float(cam.max())
    if peak <= 0:
        raise ValueError("Grad-CAM returned an empty activation map.")

    _, mask = cv2.threshold(cam, peak * 0.55, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ValueError("No Grad-CAM regions were strong enough to outline.")

    cv2.drawContours(base_image, contours, -1, (24, 126, 158), 5)
    cv2.drawContours(base_image, contours, -1, (225, 249, 255), 2)

    return base_image


def generate_gradcam_map(model, image_tensor, target_index=None):
    """Generate a Grad-CAM heatmap for a given image tensor and model."""
    model.eval()
    gradcam_input = image_tensor.detach().clone().requires_grad_(True)

    if target_index is None:
        with torch.no_grad():
            target_index = int(model(gradcam_input).argmax(dim=1).item())

    target_layers = [model.features[-1]]

    with torch.enable_grad():
        with GradCAM(
            model=model,
            target_layers=target_layers,
        ) as cam:
            grayscale_cam = cam(
                input_tensor=gradcam_input,
                targets=[ClassifierOutputTarget(int(target_index))],
            )[0]

    return grayscale_cam


def generate_gradcam(model, image_tensor, image, target_index=None):
    """Generate a Grad-CAM visualization for a given image tensor and model."""
    grayscale_cam = generate_gradcam_map(
        model=model,
        image_tensor=image_tensor,
        target_index=target_index,
    )
    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_image = np.asarray(resized_image, dtype=np.float32) / 255.0

    return show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True,
    )

def load_model():
    """Load the pre-trained MobileNet model from the checkpoint."""
    model = create_mobilenet(
        num_classes=3,
        pretrained=False
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model

def main(image_path):
    """Main function to generate Grad-CAM visualization for a given image."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    model = load_model()
    original = Image.open(image_path).convert("RGB")

    display_image = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    rgb_image = np.asarray(
        display_image,
        dtype=np.float32
    ) / 255.0

    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    input_tensor = preprocessing(
        display_image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    predicted_index = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, predicted_index].item()

    visualization = generate_gradcam(
        model=model,
        image_tensor=input_tensor,
        image=display_image,
        target_index=predicted_index,
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(visualization)
    axes[1].set_title("Grad-CAM Explanation")
    axes[1].axis("off")

    figure.suptitle(
        f"Prediction: {CLASS_NAMES[predicted_index]} "
        f"({confidence:.2%} confidence)"
    )
    plt.tight_layout()

    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_gradcam.png"

    plt.savefig(output_path, dpi=150)
    plt.show()

    print("Prediction:", CLASS_NAMES[predicted_index])
    print(f"Confidence: {confidence:.2%}")
    print("\nClass probabilities:")

    for index, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name}: "
            f"{probabilities[0, index].item():.2%}"
        )
    print("\nGrad-CAM saved to:", output_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a vehicle damage image")
    args = parser.parse_args()
    main(args.image)
