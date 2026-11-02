from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def create_image_transform(image_size=224, augment=False):
    """Create a transformation pipeline for image preprocessing."""
    pipeline = [transforms.Resize((image_size, image_size))]

    if augment:
        pipeline.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                ),
            ]
        )

    pipeline.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return transforms.Compose(pipeline)