import torch
import random
import numpy as np


def reproducability_setup(RANDOM_SEED):

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Cuda Specific
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Apple Metal Specific
    torch.mps.manual_seed(RANDOM_SEED)
