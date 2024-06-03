import torch
from torch import nn
from tqdm import tqdm
from hivit.utils import AppendLogger
import os
import json
from hivit.telegram import notify_telegram_group


def test_model(
    working_folder, training_name, model, checkpoint_path, device, test_dataloader
):
    log_file_path = os.path.join(working_folder, f"{training_name}.txt")
    logger = AppendLogger(log_file_path)
    model_path = os.path.join(working_folder, checkpoint_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_running_loss = 0
    test_labels, test_preds = [], []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for img, label in tqdm(test_dataloader, position=0, leave=True):
            img, label = img.to(device), label.to(device)
            img = img.float()  # Ensure img is float
            label = label.type(
                torch.long
            )  # Ensure label is long (for CrossEntropyLoss)

            y_pred = model(img)
            y_pred_label = torch.argmax(y_pred, dim=1)

            test_labels.extend(label.cpu().detach())
            test_preds.extend(y_pred_label.cpu().detach())

            loss = criterion(y_pred, label)
            test_running_loss += loss.item()

    test_loss = test_running_loss / len(test_dataloader)
    test_accuracy = sum(1 for x, y in zip(test_preds, test_labels) if x == y) / len(
        test_labels
    )

    logger.print(f"Test Loss: {test_loss:.4f}")
    logger.print(f"Test Accuracy: {test_accuracy:.4f}")

    results_dict = {"test_loss": test_loss, "test_accuracy": test_accuracy}

    json_file = os.path.join(working_folder, f"{training_name}_result.json")

    with open(json_file, "w") as file:
        json.dump(results_dict, file, indent=4)
    notify_telegram_group(f"TEST MODEL: {training_name}")
    notify_telegram_group(json.dumps(results_dict, indent=4))
