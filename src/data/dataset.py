
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_TO_IDX = {
    "low": 0,
    "medium": 1,
    "high": 2
}

IDX_TO_CLASS = {
    index: class_name
    for class_name, index in CLASS_TO_IDX.items()
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


class CarDamageDataset(Dataset):

    def __init__(
        self,
        root_dir,
        image_size=224,
        training=False
    ):
        self.root_dir = Path(root_dir)
        self.training = training

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}"
            )

        self.samples = []

        # Collect image paths and corresponding labels.
        for class_name, class_index in CLASS_TO_IDX.items():

            class_dir = self.root_dir / class_name

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Class directory not found: {class_dir}"
                )

            for image_path in class_dir.rglob("*"):

                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    self.samples.append(
                        (image_path, class_index)
                    )

        if not self.samples:
            raise ValueError(
                f"No images found in {self.root_dir}"
            )

        # Image transformations.
        if self.training:

            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),

                transforms.RandomHorizontalFlip(p=0.5),

                transforms.RandomRotation(degrees=10),

                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        else:

            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        print(
            f"Loaded {len(self.samples)} images "
            f"from {self.root_dir}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(
                f"Could not load image: {image_path}"
            ) from error

        image = self.transform(image)

        return image, label
