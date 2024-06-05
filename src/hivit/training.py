from .vit import Vit

import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import timeit
from tqdm import tqdm
import os
from hivit.utils import AppendLogger
from torch.optim.lr_scheduler import CosineAnnealingLR


def training_loop(
    working_folder,
    training_name,
    embedding_strategy,
    checkpoint_file_name,
    device,
    train_dataloader,
    val_dataloader,
    EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    NUM_CLASSES,
    PATCH_SIZE,
    IMAGE_SIZE,
    IN_CHANNELS,
    NUM_HEADS,
    DROPOUT,
    HIDDEN_DIM,
    ADAM_WEIGHT_DECAY,
    ADAM_BETAS,
    ACTIVATION,
    NUM_ENCODERS,
    EMBEDING_DIMENTION,
    NUM_PATCHES,
    NO_PLT_SHOW=False,
):
    log_file_path = os.path.join(working_folder, f"{training_name}.txt")
    logger = AppendLogger(log_file_path)
    model = Vit(
        embedding_strategy,
        NUM_PATCHES,
        IMAGE_SIZE,
        NUM_CLASSES,
        PATCH_SIZE,
        EMBEDING_DIMENTION,
        NUM_ENCODERS,
        NUM_HEADS,
        HIDDEN_DIM,
        DROPOUT,
        ACTIVATION,
        IN_CHANNELS,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        betas=ADAM_BETAS,
        lr=LEARNING_RATE,
        weight_decay=ADAM_WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stop = False
    start = timeit.default_timer()

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_labels, train_preds = [], []
        train_running_loss = 0

        for idx, (img, label) in enumerate(
            tqdm(train_dataloader, position=0, leave=True)
        ):
            img, label = img.to(device), label.to(device)
            img = img.float()  # Ensure img is float
            label = label.type(
                torch.long
            )  # Ensure label is long (for CrossEntropyLoss)

            y_pred = model(img)
            y_pred_label = torch.argmax(y_pred, dim=1)

            train_labels.extend(label.cpu().detach())
            train_preds.extend(y_pred_label.cpu().detach())

            loss = criterion(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

        train_loss = train_running_loss / len(train_dataloader)

        model.eval()
        val_labels, val_preds = [], []
        val_running_loss = 0

        with torch.no_grad():
            for idx, (img, label) in enumerate(
                tqdm(val_dataloader, position=0, leave=True)
            ):
                img, label = img.to(device), label.to(device)
                img = img.float()  # Ensure img is float
                label = label.type(
                    torch.long
                )  # Ensure label is long (for CrossEntropyLoss)

                y_pred = model(img)
                y_pred_label = torch.argmax(y_pred, dim=1)

                val_labels.extend(label.cpu().detach())
                val_preds.extend(y_pred_label.cpu().detach())

                loss = criterion(y_pred, label)
                val_running_loss += loss.item()

        val_loss = val_running_loss / len(val_dataloader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), os.path.join(working_folder, checkpoint_file_name)
            )
            print(
                f"Epoch {epoch + 1}: New best model saved with val_loss: {val_loss:.4f}"
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == PATIENCE:
            print("Early stopping triggered")
            early_stop = True
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(
            sum(1 for x, y in zip(train_preds, train_labels) if x == y)
            / len(train_labels)
        )
        val_accuracies.append(
            sum(1 for x, y in zip(val_preds, val_labels) if x == y) / len(val_labels)
        )

        logger.print("-" * 30)
        logger.print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        logger.print(f"Valid Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        logger.print(f"Train Accuracy EPOCH {epoch + 1}: {train_accuracies[-1]:.4f}")
        logger.print(f"Valid Accuracy EPOCH {epoch + 1}: {val_accuracies[-1]:.4f}")
        logger.print("-" * 30)

        scheduler.step()

    if not early_stop:
        logger.print("Completed all epochs without early stopping.")

    stop = timeit.default_timer()
    logger.print(f"Training Time: {stop - start:.2f}s")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle("Training and Validation Metrics")

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(train_accuracies, label="Train Accuracy")
    ax2.plot(val_accuracies, label="Validation Accuracy")
    ax2.set_title("Accuracy over epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    file_path = os.path.join(working_folder, f"{training_name}.png")

    plt.savefig(file_path)

    if not NO_PLT_SHOW:
        plt.show()
    else:
        plt.close(fig)

    model.load_state_dict(
        torch.load(os.path.join(working_folder, checkpoint_file_name))
    )

    return model
