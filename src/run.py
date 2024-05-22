# import sys

# appending a path
# sys.path.append("..")

import os
from bizim.process import train

TRAIN_RANGE = int(os.getenv("TRAIN_RANGE", "1"))
TRAIN_NAME = os.getenv("TRAIN_NAME", "TRAIN_NAME")


train(TRAIN_RANGE, 2, TRAIN_NAME)

print("Deneme sasdasdas")
