import itertools
import os


def generate_files(overrides, deault_parameters, experiment_name, output_folder):
    param_names = list(overrides.keys())
    param_values = list(overrides.values())
    experiment_folder = os.path.join(output_folder, experiment_name)
    os.makedirs(experiment_folder)

    combinations = list(itertools.product(*param_values))

    for i in range(len(combinations)):
        parameters = deault_parameters.copy()
        for j in range(len(param_names)):
            parameters[param_names[j]] = combinations[i][j]

        file_name = os.path.join(experiment_folder, f"{experiment_name}_{i:04}.sh")
        with open(file_name, "w") as file:
            file.write("#!/bin/bash\n")
            file.write(f"export EXPERIMENT_NAME={experiment_name}_{i:04}\n")
            for key, value in parameters.items():
                file.write(f"export {key}={value}\n")
                print(f"export {key}={value}", end="\n")


######################################################


deault_parameters = {
    "RANDOM_SEED": "42",
    "BATCH_SIZE": "256",
    "EPOCHS": "100",
    "PATIENCE": "200",
    "LEARNING_RATE": "1e-3",
    "LEARNING_RATE_SCHEDULER_NAME": "",
    "PATCH_SIZE": "4",
    "NUM_HEADS": "12",
    "DROPOUT": "0.3",
    "HIDDEN_DIM": "512",
    "ADAM_WEIGHT_DECAY": "0",
    "ADAM_BETAS": "(0.9, 0.999)",
    "ACTIVATION": "gelu",
    "NUM_ENCODERS": "6",
    "TRANSFORM_RANDOM_HORIZONTAL_FLIP_ENABLED": "False",
    "TRANSFORM_RANDOM_ROTATION_ENABLED": "False",
    "TRANSFORM_RANDOM_ROTATION_DEGREE": "0",
    "TRANSFORM_RANDOM_CROP_ENABLED": "False",
    "TRANSFORM_RANDOM_CROP_PADDING": "4",
    "TRANSFORM_COLOR_JITTER_ENABLED": "False",
    "TRANSFORM_COLOR_JITTER_BRIGHTNESS": "0.2",
    "TRANSFORM_COLOR_JITTER_CONTRAST": "0.2",
    "TRANSFORM_COLOR_JITTER_SATURATION": "0.2",
    "TRANSFORM_COLOR_JITTER_HUE": "0.2",
    "TRANSFORM_RANDOM_ERASING": "False",
    "EXECUTE_MODEL_LEARNING": "False",
    "EXECUTE_MODEL_HILBERT": "False",
    "EXECUTE_MODEL_NOEMBEDING": "False",
    "EXECUTE_MODEL_RESNET_EMBEDING": "True",
    "TRAINER_BOT_TOKEN": "7453991407:AAHxY8z6n8N4r84e_fYK9XQIhmLfbplNTck",
    "TRAINER_BOT_CHAT_ID": "-4236180801",
}


experiment_name = "cifar10_patch_size_comparison"
output_folder = "data/experiments"
os.makedirs(output_folder, exist_ok=True)

# Example usage:
overrides = {
    "DATASET_NAME": ["cifar10"],
    "EPOCHS": ["200"],
    "RANDOM_SEEDS": ["32"],
    "PATCH_SIZES": ["2", "4", "8", "16"],
    "BATCH_SIZES": ["128"],
    "HIDDEN_DIMS": ["768"],
    "NUM_ENCODERSS": ["8", "6"],
    "NUM_HEADSS": ["8", "6"],
}

generate_files(overrides, deault_parameters, experiment_name, output_folder)
