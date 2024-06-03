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

    def load_from_env(self):
        self.RANDOM_SEED = int(os.getenv("RANDOM_SEED", self.RANDOM_SEED))
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", self.BATCH_SIZE))
        self.EPOCHS = int(os.getenv("EPOCHS", self.EPOCHS))
        self.PATIENCE = int(os.getenv("PATIENCE", self.PATIENCE))
        self.LEARNING_RATE = float(os.getenv("LEARNING_RATE", self.LEARNING_RATE))
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

        self.ADAM_WEIGHT_DECAY = float(
            os.getenv("ADAM_WEIGHT_DECAY", self.ADAM_WEIGHT_DECAY)
        )
        self.DATASET_NAME = os.getenv("DATASET_NAME", self.DATASET_NAME)
        self.DATASET_ROOT = os.getenv("DATASET_ROOT", self.DATASET_ROOT)
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
