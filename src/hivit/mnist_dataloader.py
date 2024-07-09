from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split


def mnist_dataloader(DATASET_ROOT, BATCH_SIZE, training_transformations, IMAGE_SIZE):

    padding = (IMAGE_SIZE - 28) // 2

    transformations = []
    transformations.extend(training_transformations)
    transformations.extend(
        [
            transforms.Pad(padding),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )

    transform_train = transforms.Compose(transformations)

    transform_val_test = transforms.Compose(
        [
            transforms.Pad(padding),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )

    train_dataset = datasets.MNIST(
        root=DATASET_ROOT,
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = datasets.MNIST(
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
