import matplotlib.pyplot as plt
import numpy as np


def visualize_data(train_loader, val_loader, test_loader, num_images=2):
    def show_image(image, ax, title):
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.5071, 0.4867, 0.4408])
        std = np.array([0.2675, 0.2565, 0.2761])
        image = std * image + mean  # Unnormalize
        image = np.clip(image, 0, 1)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    # Get a batch of images from the train, val, and test loaders
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    test_iter = iter(test_loader)

    # Fetch one batch of images and labels
    train_images, train_labels = next(train_iter)
    val_images, val_labels = next(val_iter)
    test_images, test_labels = next(test_iter)

    # Plot images from each dataset
    fig, axarr = plt.subplots(num_images, 3, figsize=(15, num_images * 5))
    for i in range(num_images):
        show_image(
            train_images[i], axarr[i, 0], f"Train Image - {train_labels[i].item()}"
        )
        show_image(
            val_images[i], axarr[i, 1], f"Validation Image - {val_labels[i].item()}"
        )
        show_image(test_images[i], axarr[i, 2], f"Test Image - {test_labels[i].item()}")

    plt.tight_layout()
    plt.show()
