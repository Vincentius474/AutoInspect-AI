import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.data.dataset import CarDamageDataset
from src.data.preprocessing import create_image_transform
from src.evaluation.gradcam import draw_gradcam_contours
from src.models.mobilenet import create_mobilenet
from src.training.metrics import calculate_metrics

class PreprocessingTests(unittest.TestCase):
    """Tests for image preprocessing and dataset loading."""

    def test_transform_resizes_and_normalizes_image(self):
        image = Image.new("RGB", (40, 20), (255, 0, 0))
        tensor = create_image_transform(image_size=32)(image)
        self.assertEqual(tuple(tensor.shape), (3, 32, 32))
        self.assertAlmostEqual(
            tensor[0, 0, 0].item(),
            (1.0 - 0.485) / 0.229,
            places=5,
        )

    def test_dataset_loads_class_folders_with_shared_transform(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for class_name in ("low", "medium", "high"):
                class_directory = root / class_name
                class_directory.mkdir()
                Image.new("RGB", (24, 24), (80, 120, 160)).save(
                    class_directory / "sample.png"
                )

            dataset = CarDamageDataset(
                root,
                image_size=32,
                training=False,
            )

            self.assertEqual(len(dataset), 3)
            self.assertEqual(
                {dataset[index][1] for index in range(len(dataset))},
                {0, 1, 2},
            )
            self.assertEqual(tuple(dataset[0][0].shape), (3, 32, 32))

class ModelAndMetricsTests(unittest.TestCase):
    """Tests for model output shapes and metric calculations."""

    def test_mobilenet_returns_three_class_logits(self):
        model = create_mobilenet(num_classes=3, pretrained=False).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (1, 3))

    def test_metrics_include_per_class_scores_and_confusion_matrix(self):
        metrics = calculate_metrics([0, 1, 2], [0, 2, 2])
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["confusion_matrix"].shape, (3, 3))
        self.assertEqual(len(metrics["class_f1"]), 3)

class GradCamContourTests(unittest.TestCase):
    """Tests for Grad-CAM contour drawing functionality."""

    def test_contours_are_drawn_on_original_image(self):
        image = Image.new("RGB", (224, 224), (100, 120, 140))
        activation = np.zeros((224, 224), dtype=np.float32)
        activation[60:160, 70:160] = 1.0
        result = draw_gradcam_contours(image, activation)
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.any(result != np.asarray(image)))

    def test_empty_activation_map_is_reported(self):
        image = Image.new("RGB", (224, 224), (100, 120, 140))

        with self.assertRaisesRegex(ValueError, "empty activation map"):
            draw_gradcam_contours(
                image,
                np.zeros((224, 224), dtype=np.float32),
            )

if __name__ == "__main__":
    unittest.main()
