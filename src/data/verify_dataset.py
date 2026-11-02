from pathlib import Path
from PIL import Image, UnidentifiedImageError
import hashlib

DATA_DIR = Path("data/splits")
SPLITS = ["train", "validation", "test"]
CLASSES = ["low", "medium", "high"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

def image_hash(path):
    """Calculate an image's SHA-256 hash."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()

def main():
    """Verify the dataset structure and check for duplicate images across splits."""
    split_hashes = {}
    total_images = 0

    for split in SPLITS:
        print(f"\nChecking {split.upper()}")

        split_hashes[split] = set()

        for class_name in CLASSES:
            folder = DATA_DIR / split / class_name

            if not folder.is_dir():
                raise FileNotFoundError(folder)

            images = [
                path for path in folder.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ]

            valid = 0
            invalid = 0

            for path in images:
                try:
                    with Image.open(path) as image:
                        image.verify()

                    digest = image_hash(path)
                    split_hashes[split].add(digest)
                    valid += 1

                except (OSError, UnidentifiedImageError):
                    invalid += 1

            total_images += valid

            print(
                f"{class_name:8} "
                f"valid={valid:4} "
                f"invalid={invalid:4}"
            )

    print("\nChecking duplicate images across splits")
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:
            duplicates = split_hashes[first] & split_hashes[second]

            print(
                f"{first} vs {second}: "
                f"{len(duplicates)} exact duplicates"
            )

    print(f"\nTotal valid images: {total_images}")

if __name__ == "__main__":
    main()
