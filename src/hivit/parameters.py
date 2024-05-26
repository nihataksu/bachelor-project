import json


class Parameters:
    def __init__(self):
        self.RANDOM_SEED = 42
        self.BATCH_SIZE = 512
        self.EPOCHS = 1
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
        self.DATASET_ROOT = "./data"

    def load_from_json(self, json_file):
        with open(json_file, "r") as file:
            params = json.load(file)
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)

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


# # Example usage
# params = Parameters()
# params.load_from_json("path_to_your_json_file.json")

# # Verify the loaded parameters
# print(vars(params))
