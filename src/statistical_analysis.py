import json
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
import seaborn as sns
import matplotlib.pyplot as plt
import random

# # Generate fake data
# fake_data = {"results": []}

# datasets = ["mnist", "cifar10", "cifar100"]
# methods = ["learned_pos", "no_pos", "hilbert_pos"]

# # Generate fake data
# for dataset in datasets:
#     for method in methods:
#         for _ in range(100):  # Ensuring at least 100 runs for each method
#             entry = {
#                 "RANDOM_SEED": random.randint(0, 100),
#                 "BATCH_SIZE": 128,
#                 "EPOCHS": 200,
#                 "PATIENCE": 200,
#                 "LEARNING_RATE": 0.001,
#                 "LEARNING_RATE_SCHEDULER_NAME": "",
#                 "NUM_CLASSES": 10 if dataset == "mnist" else 100,
#                 "PATCH_SIZE": random.choice([2, 4, 8, 16]),
#                 "IMAGE_SIZE": 32,
#                 "IN_CHANNELS": 1 if dataset == "mnist" else 3,
#                 "NUM_HEADS": random.choice([4, 6, 8]),
#                 "DROPOUT": 0.3,
#                 "HIDDEN_DIM": 768,
#                 "ADAM_WEIGHT_DECAY": 0.0,
#                 "ADAM_BETAS": [0.9, 0.999],
#                 "ACTIVATION": "gelu",
#                 "NUM_ENCODERS": random.choice([4, 6, 8]),
#                 "EMBEDING_DIMENTION": 768,
#                 "NUM_PATCHES": random.choice([4, 16, 64, 256]),
#                 "DATASET_NAME": dataset,
#                 "DATASET_ROOT": "./data",
#                 "NO_PLT_SHOW": False,
#                 "GIT_HASH": "fake_hash",
#                 "GIT_IS_DIRTY": True,
#                 "GIT_COMMIT_URL": "https://github.com/fake_repo",
#                 "GIT_CHANGED_FILES": [
#                     "src/collect.py",
#                     "src/experiment2.sh",
#                     "src/generate_parameters.py",
#                     "src/hivit/parameters.py",
#                 ],
#                 "EXECUTE_MODEL_LEARNING": (
#                     "True" if method == "learned_pos" else "False"
#                 ),
#                 "EXECUTE_MODEL_HILBERT": "True" if method == "hilbert_pos" else "False",
#                 "EXECUTE_MODEL_NOEMBEDING": "True" if method == "no_pos" else "False",
#                 "EXECUTE_MODEL_RESNET_EMBEDING": "False",
#                 "EXECUTE_MODEL_RESNET_HILBERT_EMBEDING": "True",
#                 "MODEL_LEARNING_RESULT_LOSS": (
#                     random.uniform(0.01, 0.1) if method == "learned_pos" else None
#                 ),
#                 "MODEL_LEARNING_RESULT_ACCURACY": (
#                     random.uniform(0.9, 1.0) if method == "learned_pos" else None
#                 ),
#                 "MODEL_HILBERT_RESULT_LOSS": (
#                     random.uniform(0.01, 0.1) if method == "hilbert_pos" else None
#                 ),
#                 "MODEL_HILBERT_RESULT_ACCURACY": (
#                     random.uniform(0.9, 1.0) if method == "hilbert_pos" else None
#                 ),
#                 "MODEL_NOEMBEDING_RESULT_LOSS": (
#                     random.uniform(0.01, 0.1) if method == "no_pos" else None
#                 ),
#                 "MODEL_NOEMBEDING_RESULT_ACCURACY": (
#                     random.uniform(0.9, 1.0) if method == "no_pos" else None
#                 ),
#                 "TRANSFORM_RANDOM_HORIZONTAL_FLIP_ENABLED": "False",
#                 "TRANSFORM_RANDOM_ROTATION_ENABLED": "False",
#                 "TRANSFORM_RANDOM_ROTATION_DEGREE": 0,
#                 "TRANSFORM_RANDOM_CROP_ENABLED": "False",
#                 "TRANSFORM_RANDOM_CROP_PADDING": 4,
#                 "TRANSFORM_COLOR_JITTER_ENABLED": "False",
#                 "TRANSFORM_COLOR_JITTER_BRIGHTNESS": 0.2,
#                 "TRANSFORM_COLOR_JITTER_CONTRAST": 0.2,
#                 "TRANSFORM_COLOR_JITTER_SATURATION": 0.2,
#                 "TRANSFORM_COLOR_JITTER_HUE": 0.2,
#                 "TRANSFORM_RANDOM_ERASING": "False",
#                 "DEVICE": "cuda",
#                 "SLURM_JOB_ID": "",
#                 "SLURM_JOB_NUM_NODES": "",
#                 "SLURM_NTASKS": "",
#                 "SLURM_NTASKS_PER_NODE": "",
#                 "SLURM_TASKS_PER_NODE": "",
#                 "SLURM_CPUS_PER_TASK": "",
#                 "SLURM_JOB_CPUS_ON_NODE": "",
#                 "SLURM_NTASKS_PER_CORE": "",
#             }
#             fake_data["results"].append(entry)

# # Save the generated fake data to a JSON file
# file_path = "/Users/nihat/repos/bachelor-project/src/test_stats.json"
# with open(file_path, "w") as file:
#     json.dump(fake_data, file, indent=4)


# Load JSON data
file_path = "/Users/nihat/repos/bachelor-project/src/2024-07-26-final-results-omer.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Initialize result lists
results = {
    "mnist_learned_pos": [],
    "mnist_no_pos": [],
    "mnist_hilbert_pos": [],
    "cifar10_learned_pos": [],
    "cifar10_no_pos": [],
    "cifar10_hilbert_pos": [],
    "cifar100_learned_pos": [],
    "cifar100_no_pos": [],
    "cifar100_hilbert_pos": [],
}

# Collect results
for entry in data["results"]:
    dataset = entry["DATASET_NAME"]
    if (
        entry["EXECUTE_MODEL_LEARNING"] == "True"
        and entry["MODEL_LEARNING_RESULT_ACCURACY"] is not None
    ):
        results[f"{dataset}_learned_pos"].append(
            entry["MODEL_LEARNING_RESULT_ACCURACY"]
        )
    if (
        entry["EXECUTE_MODEL_NOEMBEDING"] == "True"
        and entry["MODEL_NOEMBEDING_RESULT_ACCURACY"] is not None
    ):
        results[f"{dataset}_no_pos"].append(entry["MODEL_NOEMBEDING_RESULT_ACCURACY"])
    if (
        entry["EXECUTE_MODEL_HILBERT"] == "True"
        and entry["MODEL_HILBERT_RESULT_ACCURACY"] is not None
    ):
        results[f"{dataset}_hilbert_pos"].append(entry["MODEL_HILBERT_RESULT_ACCURACY"])

# Debug: Print collected results
print("Collected results:")
for key, value in results.items():
    print(f"{key}: {len(value)} entries")

# Prepare data for ANOVA
anova_data = []
for key, values in results.items():
    method = key.split("_")[-2] + "_" + key.split("_")[-1]
    dataset = "_".join(key.split("_")[:-2])
    for value in values:
        anova_data.append({"dataset": dataset, "method": method, "accuracy": value})

df = pd.DataFrame(anova_data)

# Debug: Print ANOVA dataframe
print("ANOVA DataFrame:")
print(df.head())


# Function to perform ANOVA and Tukey's HSD test
def perform_anova(dataset):
    subset = df[df["dataset"] == dataset]
    unique_methods = subset["method"].unique()

    # Debug: Print unique methods for the dataset
    print(f"Dataset: {dataset}, Unique methods: {unique_methods}")

    if len(unique_methods) < 2:
        print(f"Not enough methods for ANOVA in dataset {dataset}. Skipping...")
        return

    model = ols("accuracy ~ C(method)", data=subset).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f"ANOVA results for {dataset} dataset:")
    print(anova_table)

    if anova_table["PR(>F)"].iloc[0] < 0.05:
        print("ANOVA is significant, proceeding with post-hoc tests.")
        comp = mc.MultiComparison(subset["accuracy"], subset["method"])
        post_hoc_res = comp.tukeyhsd()
        print(post_hoc_res)
    else:
        print("ANOVA is not significant, no further tests needed.")


# Perform ANOVA for each dataset
datasets = ["mnist", "cifar10", "cifar100"]
for dataset in datasets:
    perform_anova(dataset)


# Define the plotting function
def plot_violin_plots(df, datasets, y_min=None, y_max=None):
    for dataset in datasets:
        plt.figure(figsize=(14, 8))
        sns.violinplot(
            x="method",
            y="accuracy",
            data=df[df["dataset"] == dataset],
            inner="quart",
            palette="muted",
        )
        plt.title(f"Accuracy Distribution for {dataset}")
        plt.xlabel("Method")
        plt.ylabel("Accuracy")
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        plt.show()


# Example usage with control over y-axis range
plot_violin_plots(df, ["mnist", "cifar10", "cifar100"], y_min=None, y_max=None)
