import json
import os
from .git import get_current_git_commit
from .git import is_git_repo_dirty
from .git import get_current_commit_url
from .git import get_changed_files


class Parameters:
    def __init__(self):
        self.RANDOM_SEED = 42
        self.BATCH_SIZE = 512
        self.EPOCHS = 2
        self.PATIENCE = 20
        self.LEARNING_RATE = 1e-3
        self.LEARNING_RATE_SCHEDULER_NAME = ""
        self.NUM_CLASSES = 10
        self.PATCH_SIZE = 4
        self.IMAGE_SIZE = 32
        self.IN_CHANNELS = 3
        self.NUM_HEADS = 12
        self.DROPOUT = 0.3
        self.HIDDEN_DIM = 1024
        self.ADAM_WEIGHT_DECAY = 0
        self.ADAM_BETAS = (0.9, 0.999)
        self.ACTIVATION = "gelu"
        self.NUM_ENCODERS = 12
        self.EMBEDING_DIMENTION = (self.PATCH_SIZE**2) * self.IN_CHANNELS
        self.NUM_PATCHES = (self.IMAGE_SIZE // self.PATCH_SIZE) ** 2
        self.DATASET_NAME = ""
        self.DATASET_ROOT = "./data"
        self.NO_PLT_SHOW = False
        self.GIT_HASH = ""
        self.GIT_IS_DIRTY = ""
        self.GIT_COMMIT_URL = ""
        self.GIT_CHANGED_FILES = []
        self.EXECUTE_MODEL_LEARNING = "True"
        self.EXECUTE_MODEL_HILBERT = "True"
        self.EXECUTE_MODEL_NOEMBEDING = "True"
        self.EXECUTE_MODEL_RESNET_EMBEDING = "True"
        self.EXECUTE_MODEL_RESNET_HILBERT_EMBEDING = "True"
        self.MODEL_LEARNING_RESULT_LOSS = None
        self.MODEL_LEARNING_RESULT_ACCURACY = None
        self.MODEL_HILBERT_RESULT_LOSS = None
        self.MODEL_HILBERT_RESULT_ACCURACY = None
        self.MODEL_NOEMBEDING_RESULT_LOSS = None
        self.MODEL_NOEMBEDING_RESULT_ACCURACY = None
        self.TRANSFORM_RANDOM_HORIZONTAL_FLIP_ENABLED = "False"
        self.TRANSFORM_RANDOM_ROTATION_ENABLED = "False"
        self.TRANSFORM_RANDOM_ROTATION_DEGREE = "0"
        self.TRANSFORM_RANDOM_CROP_ENABLED = "False"
        self.TRANSFORM_RANDOM_CROP_PADDING = "4"
        self.TRANSFORM_COLOR_JITTER_ENABLED = "False"
        self.TRANSFORM_COLOR_JITTER_BRIGHTNESS = "0.2"
        self.TRANSFORM_COLOR_JITTER_CONTRAST = "0.2"
        self.TRANSFORM_COLOR_JITTER_SATURATION = "0.2"
        self.TRANSFORM_COLOR_JITTER_HUE = "0.2"
        self.TRANSFORM_RANDOM_ERASING = "False"
        self.DEVICE = ""
        self.SLURM_JOB_ID = ""
        self.SLURM_JOB_NUM_NODES = ""
        self.SLURM_NTASKS = ""
        self.SLURM_NTASKS_PER_NODE = ""
        self.SLURM_TASKS_PER_NODE = ""
        self.SLURM_CPUS_PER_TASK = ""
        self.SLURM_JOB_CPUS_ON_NODE = ""
        self.SLURM_NTASKS_PER_CORE = ""

    def load_from_env(self):
        self.NO_PLT_SHOW = os.getenv("NO_PLT_SHOW", self.NO_PLT_SHOW)
        self.RANDOM_SEED = int(os.getenv("RANDOM_SEED", self.RANDOM_SEED))
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", self.BATCH_SIZE))
        self.EPOCHS = int(os.getenv("EPOCHS", self.EPOCHS))
        self.PATIENCE = int(os.getenv("PATIENCE", self.PATIENCE))
        self.LEARNING_RATE = float(os.getenv("LEARNING_RATE", self.LEARNING_RATE))
        self.LEARNING_RATE_SCHEDULER_NAME = os.getenv(
            "LEARNING_RATE_SCHEDULER_NAME", self.LEARNING_RATE_SCHEDULER_NAME
        )
        self.NUM_CLASSES = int(os.getenv("NUM_CLASSES", self.NUM_CLASSES))
        self.PATCH_SIZE = int(os.getenv("PATCH_SIZE", self.PATCH_SIZE))
        self.IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", self.IMAGE_SIZE))
        self.IN_CHANNELS = int(os.getenv("IN_CHANNELS", self.IN_CHANNELS))
        self.NUM_HEADS = int(os.getenv("NUM_HEADS", self.NUM_HEADS))
        self.NUM_ENCODERS = int(os.getenv("NUM_ENCODERS", self.NUM_ENCODERS))
        self.DROPOUT = float(os.getenv("DROPOUT", self.DROPOUT))
        self.HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", self.HIDDEN_DIM))
        self.EXECUTE_MODEL_LEARNING = os.getenv(
            "EXECUTE_MODEL_LEARNING", self.EXECUTE_MODEL_LEARNING
        )
        self.EXECUTE_MODEL_HILBERT = os.getenv(
            "EXECUTE_MODEL_HILBERT", self.EXECUTE_MODEL_HILBERT
        )
        self.EXECUTE_MODEL_NOEMBEDING = os.getenv(
            "EXECUTE_MODEL_NOEMBEDING", self.EXECUTE_MODEL_NOEMBEDING
        )

        self.EXECUTE_MODEL_RESNET_EMBEDING = os.getenv(
            "EXECUTE_MODEL_RESNET_EMBEDING",
            self.EXECUTE_MODEL_RESNET_EMBEDING,
        )

        self.EXECUTE_MODEL_RESNET_HILBERT_EMBEDING = os.getenv(
            "EXECUTE_MODEL_RESNET_HILBERT_EMBEDING",
            self.EXECUTE_MODEL_RESNET_HILBERT_EMBEDING,
        )

        self.ADAM_WEIGHT_DECAY = float(
            os.getenv("ADAM_WEIGHT_DECAY", self.ADAM_WEIGHT_DECAY)
        )
        self.DATASET_NAME = os.getenv("DATASET_NAME", self.DATASET_NAME)
        self.DATASET_ROOT = os.getenv("DATASET_ROOT", self.DATASET_ROOT)

        self.TRANSFORM_RANDOM_HORIZONTAL_FLIP_ENABLED = os.getenv(
            "TRANSFORM_RANDOM_HORIZONTAL_FLIP_ENABLED",
            self.TRANSFORM_RANDOM_HORIZONTAL_FLIP_ENABLED,
        )
        self.TRANSFORM_RANDOM_ROTATION_ENABLED = os.getenv(
            "TRANSFORM_RANDOM_ROTATION_ENABLED",
            self.TRANSFORM_RANDOM_ROTATION_ENABLED,
        )

        self.TRANSFORM_RANDOM_ROTATION_DEGREE = int(
            os.getenv(
                "TRANSFORM_RANDOM_ROTATION_DEGREE",
                self.TRANSFORM_RANDOM_ROTATION_DEGREE,
            )
        )

        self.TRANSFORM_RANDOM_CROP_ENABLED = os.getenv(
            "TRANSFORM_RANDOM_CROP_ENABLED",
            self.TRANSFORM_RANDOM_CROP_ENABLED,
        )

        self.TRANSFORM_RANDOM_CROP_PADDING = int(
            os.getenv(
                "TRANSFORM_RANDOM_CROP_PADDING",
                self.TRANSFORM_RANDOM_CROP_PADDING,
            )
        )

        self.TRANSFORM_COLOR_JITTER_ENABLED = os.getenv(
            "TRANSFORM_COLOR_JITTER_ENABLED",
            self.TRANSFORM_COLOR_JITTER_ENABLED,
        )

        self.TRANSFORM_COLOR_JITTER_BRIGHTNESS = float(
            os.getenv(
                "TRANSFORM_COLOR_JITTER_BRIGHTNESS",
                self.TRANSFORM_COLOR_JITTER_BRIGHTNESS,
            )
        )

        self.TRANSFORM_COLOR_JITTER_CONTRAST = float(
            os.getenv(
                "TRANSFORM_COLOR_JITTER_CONTRAST",
                self.TRANSFORM_COLOR_JITTER_CONTRAST,
            )
        )

        self.TRANSFORM_COLOR_JITTER_SATURATION = float(
            os.getenv(
                "TRANSFORM_COLOR_JITTER_SATURATION",
                self.TRANSFORM_COLOR_JITTER_SATURATION,
            )
        )

        self.TRANSFORM_COLOR_JITTER_HUE = float(
            os.getenv(
                "TRANSFORM_COLOR_JITTER_HUE",
                self.TRANSFORM_COLOR_JITTER_HUE,
            )
        )

        self.TRANSFORM_RANDOM_ERASING = os.getenv(
            "TRANSFORM_RANDOM_ERASING",
            self.TRANSFORM_RANDOM_ERASING,
        )

        self.DEVICE = os.getenv("DEVICE", self.DEVICE)
        self.SLURM_JOB_ID = os.getenv("SLURM_JOB_ID", self.SLURM_JOB_ID)
        self.SLURM_JOB_NUM_NODES = os.getenv(
            "SLURM_JOB_NUM_NODES", self.SLURM_JOB_NUM_NODES
        )
        self.SLURM_NTASKS = os.getenv("SLURM_NTASKS", self.SLURM_NTASKS)
        self.SLURM_NTASKS_PER_NODE = os.getenv(
            "SLURM_NTASKS_PER_NODE", self.SLURM_NTASKS_PER_NODE
        )
        self.SLURM_TASKS_PER_NODE = os.getenv(
            "SLURM_TASKS_PER_NODE", self.SLURM_TASKS_PER_NODE
        )
        self.SLURM_CPUS_PER_TASK = os.getenv(
            "SLURM_CPUS_PER_TASK", self.SLURM_CPUS_PER_TASK
        )
        self.SLURM_JOB_CPUS_ON_NODE = os.getenv(
            "SLURM_JOB_CPUS_ON_NODE", self.SLURM_JOB_CPUS_ON_NODE
        )
        self.SLURM_NTASKS_PER_CORE = os.getenv(
            "SLURM_NTASKS_PER_CORE", self.SLURM_NTASKS_PER_CORE
        )
        self.validate()

    def validate(self):
        self.EMBEDING_DIMENTION = (self.PATCH_SIZE**2) * self.IN_CHANNELS
        self.NUM_PATCHES = (self.IMAGE_SIZE // self.PATCH_SIZE) ** 2
        self.GIT_HASH = get_current_git_commit()
        self.GIT_IS_DIRTY = is_git_repo_dirty()
        self.GIT_COMMIT_URL = get_current_commit_url()
        self.GIT_CHANGED_FILES = get_changed_files()

    def load_from_json(self, json_file):
        with open(json_file, "r") as file:
            params = json.load(file)
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        self.validate()

    def save_to_json(self, json_file):
        params = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("__")
        }
        with open(json_file, "w") as file:
            json.dump(params, file, indent=4)

    def print(self):
        params = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("__")
        }
        print(params)

    def dictionary(self):
        params = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("__")
        }
        return params

    def to_json(self):
        params = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("__")
        }
        return json.dumps(params, indent=4)


# # Example usage
# params = Parameters()
# params.load_from_json("path_to_your_json_file.json")

# # Verify the loaded parameters
# print(vars(params))
