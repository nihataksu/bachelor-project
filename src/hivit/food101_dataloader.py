from torchvision import transforms, datasets
from hivit.cut_out import Cutout
from torch.utils.data import DataLoader, random_split


def food101_dataloader(DATASET_ROOT, BATCH_SIZE):
    transform_train = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomCrop(128, padding=4),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"
            ),
        ]
    )

    transform_val_test = transforms.Compose(
        [
            transforms.Resize((128, 128)),  # Ensure the image size is 256x256
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.Food101(
        root=DATASET_ROOT,
        split="train",
        download=True,
        transform=transform_train,
    )
    test_dataset = datasets.Food101(
        root=DATASET_ROOT,
        split="test",
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
