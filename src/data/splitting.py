from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/splits")

CLASS_MAPPING = {
    "01-minor": "low",
    "02-moderate": "medium",
    "03-severe": "high",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

TEST_SIZE = 0.15
RANDOM_STATE = 42

def collect_images(directory: Path):
    """Collect images from the original dataset and map them to low/medium/high classes."""
    images = []
    for original_class, model_class in CLASS_MAPPING.items():
        class_dir = directory / original_class

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class directory not found: {class_dir}"
            )

        class_images = [
            path for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not class_images:
            raise ValueError(f"No images found in {class_dir}")

        images.extend(
            (image_path, model_class)
            for image_path in class_images
        )

    return images


def copy_images(images, destination: Path):
    """Copy images into low/medium/high destination folders."""
    for source_path, label in images:
        target_dir = destination / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name
        counter = 1
        while target_path.exists():
            target_path = target_dir / (
                f"{source_path.stem}_{counter}{source_path.suffix}"
            )
            counter += 1

        shutil.copy2(source_path, target_path)

def main():
    """Main function to split the dataset."""
    original_train_dir = RAW_DIR / "training"
    original_validation_dir = RAW_DIR / "validation"

    if not original_train_dir.is_dir():
        raise FileNotFoundError(original_train_dir)

    if not original_validation_dir.is_dir():
        raise FileNotFoundError(original_validation_dir)

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} already contains files. "
            "Move or rename it before generating new splits."
        )

    train_images = collect_images(original_train_dir)
    validation_images = collect_images(original_validation_dir)

    train_labels = [label for _, label in train_images]

    train_subset, test_subset = train_test_split(
        train_images,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_labels,
    )

    copy_images(
        train_subset,
        OUTPUT_DIR / "train"
    )

    copy_images(
        validation_images,
        OUTPUT_DIR / "validation"
    )

    copy_images(
        test_subset,
        OUTPUT_DIR / "test"
    )

    print("\nDataset preparation complete!")
    for split in ("train", "validation", "test"):
        print(f"\n{split.upper()}")

        for label in ("low", "medium", "high"):
            folder = OUTPUT_DIR / split / label

            count = sum(
                1 for path in folder.iterdir()
                if path.is_file()
            ) if folder.exists() else 0

            print(f"  {label}: {count} images")

if __name__ == "__main__":
    main()
