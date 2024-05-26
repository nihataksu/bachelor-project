from torchvision import transforms, datasets
from hivit.cut_out import Cutout
from torch.utils.data import DataLoader, random_split


def cifar10_dataloader(DATASET_ROOT, BATCH_SIZE):
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=8),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_val_test = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # Ensure the image size is 32x32
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=DATASET_ROOT,
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = datasets.CIFAR10(
        root=DATASET_ROOT,
        train=False,
        download=True,
        transform=transform_val_test,
    )

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader