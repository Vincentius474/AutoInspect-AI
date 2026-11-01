from torch.utils.data import DataLoader
from src.data.dataset import CarDamageDataset

def create_dataloaders( data_dir="data/splits", batch_size=16, image_size=224, num_workers=0):
    
    data_dir = data_dir

    train_dataset = CarDamageDataset(
        root_dir=f"{data_dir}/train",
        image_size=image_size,
        training=True
    )

    validation_dataset = CarDamageDataset(
        root_dir=f"{data_dir}/validation",
        image_size=image_size,
        training=False
    )

    test_dataset = CarDamageDataset(
        root_dir=f"{data_dir}/test",
        image_size=image_size,
        training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, validation_loader, test_loader
