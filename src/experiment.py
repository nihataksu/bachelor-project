from hivit.parameters import Parameters
from hivit.hilbert_potitional_embedding import PatchEmbeddingHilbertPositionalEmbedding
from hivit.no_positional_embedding import PatchEmbeddingNoPositionalEmbedding
from hivit.learned_positional_embedding import PatchEmbeddingLearnedPositionalEmbedding
from hivit.resnet_learned_embedding import (
    PatchEmbeddingResnetLearnedPositionalEmbedding,
)
from hivit.resnet_hilbert_embedding import (
    PatchEmbeddingResnetHilbertPositionalEmbedding,
)
from hivit.training import training_loop
from hivit.test_model import test_model
from hivit.telegram import notify_telegram_group
from hivit.telegram import send_photo_telegram_group
from datetime import datetime
from hivit.cifar10_dataloader import cifar10_dataloader
from hivit.cifar10_no_augmnet_dataloader import cifar10_no_augmnet_dataloader
from hivit.cifar100_dataloader import cifar100_dataloader
from hivit.food101_dataloader import food101_dataloader
from hivit.fashion_mnist_dataloader import fashion_mnist_dataloader
from hivit.mnist_dataloader import mnist_dataloader

import torch
import os
import sys


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    print("Metal or CUDA is not found!")
    sys.exit(1)

print(device)

# Create a working folder based on time
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
working_root = os.getenv("WORKING_ROOT", "experiment_results")
working_folder = os.path.join(working_root, f"experiment_{current_time}")
os.makedirs(working_folder, exist_ok=True)

# Get the parameters
parameters = Parameters()
parameters.load_from_env()

NO_PLT_SHOW = os.getenv("NO_PLT_SHOW") == "True"
parameters.print()

parameters.save_to_json(os.path.join(working_folder, "parameters.json"))

notify_telegram_group(f"PARAMETERS:\n{parameters.to_json()}")

match parameters.DATASET_NAME:
    case "cifar10":
        # Get the data loaders for train validation and test
        train_dataloader, val_dataloader, test_dataloader = cifar10_dataloader(
            parameters.DATASET_ROOT, parameters.BATCH_SIZE
        )
    case "cifar10_no_augment":
        # Get the data loaders for train validation and test
        train_dataloader, val_dataloader, test_dataloader = (
            cifar10_no_augmnet_dataloader(
                parameters.DATASET_ROOT, parameters.BATCH_SIZE
            )
        )
    case "cifar100":
        # Get the data loaders for train validation and test
        train_dataloader, val_dataloader, test_dataloader = cifar100_dataloader(
            parameters.DATASET_ROOT, parameters.BATCH_SIZE
        )
    case "food101":
        # Get the data loaders for train validation and test
        train_dataloader, val_dataloader, test_dataloader = food101_dataloader(
            parameters.DATASET_ROOT, parameters.BATCH_SIZE
        )
    case "mnist":
        # Get the data loaders for train validation and test
        train_dataloader, val_dataloader, test_dataloader = mnist_dataloader(
            parameters.DATASET_ROOT, parameters.BATCH_SIZE
        )
    case "fashion_mnist":
        # Get the data loaders for train validation and test
        train_dataloader, val_dataloader, test_dataloader = fashion_mnist_dataloader(
            parameters.DATASET_ROOT, parameters.BATCH_SIZE
        )
    case _:
        print("ERROR: DATASET_NAME required")
        sys.exit(-1)

if parameters.EXECUTE_MODEL_LEARNING == "True":
    # Instantiate learned positional emmbedding strategy
    print("Instantiate learned positional emmbedding strategy")
    learned_postional_embedding = PatchEmbeddingLearnedPositionalEmbedding(
        parameters.EMBEDING_DIMENTION,
        parameters.PATCH_SIZE,
        parameters.NUM_PATCHES,
        parameters.DROPOUT,
        parameters.IN_CHANNELS,
    )

    # Train the model with learned positional emmbedding
    print("Train the model with learned positional emmbedding")
    model_learned = training_loop(
        working_folder=working_folder,
        training_name="learned_training",
        embedding_strategy=learned_postional_embedding,
        checkpoint_file_name="learned_training.pt",
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        EPOCHS=parameters.EPOCHS,
        PATIENCE=parameters.PATIENCE,
        LEARNING_RATE=parameters.LEARNING_RATE,
        LEARNING_RATE_SCHEDULER_NAME=parameters.LEARNING_RATE_SCHEDULER_NAME,
        NUM_CLASSES=parameters.NUM_CLASSES,
        PATCH_SIZE=parameters.PATCH_SIZE,
        IMAGE_SIZE=parameters.IMAGE_SIZE,
        IN_CHANNELS=parameters.IN_CHANNELS,
        NUM_HEADS=parameters.NUM_HEADS,
        DROPOUT=parameters.DROPOUT,
        HIDDEN_DIM=parameters.HIDDEN_DIM,
        ADAM_WEIGHT_DECAY=parameters.ADAM_WEIGHT_DECAY,
        ADAM_BETAS=parameters.ADAM_BETAS,
        ACTIVATION=parameters.ACTIVATION,
        NUM_ENCODERS=parameters.NUM_ENCODERS,
        EMBEDING_DIMENTION=parameters.EMBEDING_DIMENTION,
        NUM_PATCHES=parameters.NUM_PATCHES,
        NO_PLT_SHOW=NO_PLT_SHOW,
    )

    # Test model with learned positional emmbedding
    print("Test model with learned positional emmbedding")
    test_model(
        working_folder=working_folder,
        training_name="learned_test",
        model=model_learned,
        checkpoint_path="learned_training.pt",
        device=device,
        test_dataloader=test_dataloader,
    )

    send_photo_telegram_group(
        os.path.join(working_folder, "learned_training.png"), "learned_training"
    )

if parameters.EXECUTE_MODEL_HILBERT == "True":
    # Instantiate hilbert positional emmbedding strategy
    print("Instantiate hilbert positional emmbedding strategy")
    hilbert_embedding = PatchEmbeddingHilbertPositionalEmbedding(
        parameters.EMBEDING_DIMENTION,
        parameters.PATCH_SIZE,
        parameters.NUM_PATCHES,
        parameters.DROPOUT,
        parameters.IN_CHANNELS,
        parameters.IMAGE_SIZE,
    )
    # Train the model with hilbert positional emmbedding
    print("Train the model with hilbert positional emmbedding")
    model_hilbert = training_loop(
        working_folder=working_folder,
        training_name="hilbert_training",
        embedding_strategy=hilbert_embedding,
        checkpoint_file_name="hilbert_training.pt",
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        EPOCHS=parameters.EPOCHS,
        PATIENCE=parameters.PATIENCE,
        LEARNING_RATE=parameters.LEARNING_RATE,
        LEARNING_RATE_SCHEDULER_NAME=parameters.LEARNING_RATE_SCHEDULER_NAME,
        NUM_CLASSES=parameters.NUM_CLASSES,
        PATCH_SIZE=parameters.PATCH_SIZE,
        IMAGE_SIZE=parameters.IMAGE_SIZE,
        IN_CHANNELS=parameters.IN_CHANNELS,
        NUM_HEADS=parameters.NUM_HEADS,
        DROPOUT=parameters.DROPOUT,
        HIDDEN_DIM=parameters.HIDDEN_DIM,
        ADAM_WEIGHT_DECAY=parameters.ADAM_WEIGHT_DECAY,
        ADAM_BETAS=parameters.ADAM_BETAS,
        ACTIVATION=parameters.ACTIVATION,
        NUM_ENCODERS=parameters.NUM_ENCODERS,
        EMBEDING_DIMENTION=parameters.EMBEDING_DIMENTION,
        NUM_PATCHES=parameters.NUM_PATCHES,
        NO_PLT_SHOW=NO_PLT_SHOW,
    )
    # Test model with hilbert positional emmbedding
    print("Test model with hilbert positional emmbedding")
    test_model(
        working_folder=working_folder,
        training_name="hilbert_test",
        model=model_hilbert,
        checkpoint_path="hilbert_training.pt",
        device=device,
        test_dataloader=test_dataloader,
    )

    send_photo_telegram_group(
        os.path.join(working_folder, "hilbert_training.png"), "hilbert_training"
    )

if parameters.EXECUTE_MODEL_NOEMBEDING == "True":
    # Instantiate no positional emmbedding strategy
    print("Instantiate no positional emmbedding strategy")
    no_postional_embedding = PatchEmbeddingNoPositionalEmbedding(
        parameters.EMBEDING_DIMENTION,
        parameters.PATCH_SIZE,
        parameters.NUM_PATCHES,
        parameters.DROPOUT,
        parameters.IN_CHANNELS,
    )

    # Train the model with no positional emmbedding
    print("Train the model with no positional emmbedding")
    model_no_embedding = training_loop(
        working_folder=working_folder,
        training_name="no_positional_embedding_training",
        embedding_strategy=no_postional_embedding,
        checkpoint_file_name="no_positional_embedding_training.pt",
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        EPOCHS=parameters.EPOCHS,
        PATIENCE=parameters.PATIENCE,
        LEARNING_RATE=parameters.LEARNING_RATE,
        LEARNING_RATE_SCHEDULER_NAME=parameters.LEARNING_RATE_SCHEDULER_NAME,
        NUM_CLASSES=parameters.NUM_CLASSES,
        PATCH_SIZE=parameters.PATCH_SIZE,
        IMAGE_SIZE=parameters.IMAGE_SIZE,
        IN_CHANNELS=parameters.IN_CHANNELS,
        NUM_HEADS=parameters.NUM_HEADS,
        DROPOUT=parameters.DROPOUT,
        HIDDEN_DIM=parameters.HIDDEN_DIM,
        ADAM_WEIGHT_DECAY=parameters.ADAM_WEIGHT_DECAY,
        ADAM_BETAS=parameters.ADAM_BETAS,
        ACTIVATION=parameters.ACTIVATION,
        NUM_ENCODERS=parameters.NUM_ENCODERS,
        EMBEDING_DIMENTION=parameters.EMBEDING_DIMENTION,
        NUM_PATCHES=parameters.NUM_PATCHES,
        NO_PLT_SHOW=NO_PLT_SHOW,
    )

    # Test model with no positional emmbedding
    print("Test model with no positional emmbedding")
    test_model(
        working_folder=working_folder,
        training_name="no_positional_embedding_test",
        model=model_no_embedding,
        checkpoint_path="no_positional_embedding_training.pt",
        device=device,
        test_dataloader=test_dataloader,
    )

    send_photo_telegram_group(
        os.path.join(working_folder, "no_positional_embedding_training.png"),
        "no_positional_embedding_training",
    )


if parameters.EXECUTE_MODEL_RESNET_EMBEDING == "True":
    # Instantiate resnet-18 positional emmbedding strategy
    print("Instantiate resnet-18 positional emmbedding strategy")
    no_postional_embedding = PatchEmbeddingResnetLearnedPositionalEmbedding(
        parameters.EMBEDING_DIMENTION,
        parameters.PATCH_SIZE,
        parameters.NUM_PATCHES,
        parameters.DROPOUT,
        parameters.IN_CHANNELS,
        parameters.IMAGE_SIZE,
    )

    # Train the model with no positional emmbedding
    print("Train the model with resnet-18 positional emmbedding")
    model_no_embedding = training_loop(
        working_folder=working_folder,
        training_name="resnet-18_embedding_training",
        embedding_strategy=no_postional_embedding,
        checkpoint_file_name="resnet-18_embedding_training.pt",
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        EPOCHS=parameters.EPOCHS,
        PATIENCE=parameters.PATIENCE,
        LEARNING_RATE=parameters.LEARNING_RATE,
        LEARNING_RATE_SCHEDULER_NAME=parameters.LEARNING_RATE_SCHEDULER_NAME,
        NUM_CLASSES=parameters.NUM_CLASSES,
        PATCH_SIZE=parameters.PATCH_SIZE,
        IMAGE_SIZE=parameters.IMAGE_SIZE,
        IN_CHANNELS=parameters.IN_CHANNELS,
        NUM_HEADS=parameters.NUM_HEADS,
        DROPOUT=parameters.DROPOUT,
        HIDDEN_DIM=parameters.HIDDEN_DIM,
        ADAM_WEIGHT_DECAY=parameters.ADAM_WEIGHT_DECAY,
        ADAM_BETAS=parameters.ADAM_BETAS,
        ACTIVATION=parameters.ACTIVATION,
        NUM_ENCODERS=parameters.NUM_ENCODERS,
        EMBEDING_DIMENTION=parameters.EMBEDING_DIMENTION,
        NUM_PATCHES=parameters.NUM_PATCHES,
        NO_PLT_SHOW=NO_PLT_SHOW,
    )

    # Test model with no positional emmbedding
    print("Test model with resnet-18 emmbedding")
    test_model(
        working_folder=working_folder,
        training_name="resnet_18_embedding_test",
        model=model_no_embedding,
        checkpoint_path="resnet_18_embedding_training.pt",
        device=device,
        test_dataloader=test_dataloader,
    )

    send_photo_telegram_group(
        os.path.join(working_folder, "resnet_18_embedding_training.png"),
        "resnet_18_embedding_training",
    )


if parameters.EXECUTE_MODEL_RESNET_HILBERT_EMBEDING == "True":
    # Instantiate resnet-18 positional emmbedding strategy
    print("Instantiate resnet-50 hilbert positional emmbedding strategy")
    no_postional_embedding = PatchEmbeddingResnetHilbertPositionalEmbedding(
        parameters.EMBEDING_DIMENTION,
        parameters.PATCH_SIZE,
        parameters.NUM_PATCHES,
        parameters.DROPOUT,
        parameters.IN_CHANNELS,
        parameters.IMAGE_SIZE,
    )

    # Train the model with no positional emmbedding
    print("Train the model with resnet-50 hilbert positional emmbedding")
    model_no_embedding = training_loop(
        working_folder=working_folder,
        training_name="resnet_50_hilbert_embedding_training",
        embedding_strategy=no_postional_embedding,
        checkpoint_file_name="resnet_50_hilbert_embedding_training.pt",
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        EPOCHS=parameters.EPOCHS,
        PATIENCE=parameters.PATIENCE,
        LEARNING_RATE=parameters.LEARNING_RATE,
        LEARNING_RATE_SCHEDULER_NAME=parameters.LEARNING_RATE_SCHEDULER_NAME,
        NUM_CLASSES=parameters.NUM_CLASSES,
        PATCH_SIZE=parameters.PATCH_SIZE,
        IMAGE_SIZE=parameters.IMAGE_SIZE,
        IN_CHANNELS=parameters.IN_CHANNELS,
        NUM_HEADS=parameters.NUM_HEADS,
        DROPOUT=parameters.DROPOUT,
        HIDDEN_DIM=parameters.HIDDEN_DIM,
        ADAM_WEIGHT_DECAY=parameters.ADAM_WEIGHT_DECAY,
        ADAM_BETAS=parameters.ADAM_BETAS,
        ACTIVATION=parameters.ACTIVATION,
        NUM_ENCODERS=parameters.NUM_ENCODERS,
        EMBEDING_DIMENTION=parameters.EMBEDING_DIMENTION,
        NUM_PATCHES=parameters.NUM_PATCHES,
        NO_PLT_SHOW=NO_PLT_SHOW,
    )

    # Test model with no positional emmbedding
    print("Test model with resnet-50 hilbert emmbedding")
    test_model(
        working_folder=working_folder,
        training_name="resnet_50_hilbert_embedding_test",
        model=model_no_embedding,
        checkpoint_path="resnet_50_hilbert_embedding_training.pt",
        device=device,
        test_dataloader=test_dataloader,
    )

    send_photo_telegram_group(
        os.path.join(working_folder, "resnet_50_hilbert_embedding_training.png"),
        "resnet_50_hilbert_embedding_training",
    )
