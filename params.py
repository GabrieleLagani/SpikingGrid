import torch

# Command line parameters
DEFAULT_CONFIG = "mnist01_240n_bio"
MODE_TRN = 'train'
MODE_TST = 'test'
DEFAULT_MODE = MODE_TST
DEFAULT_SEED = 0

# Environment parameters
RESULT_FOLDER = "results"
N_WORKERS = 0
GPU = torch.cuda.is_available()
DEVICE = "cuda:0" if GPU else "cpu"

# Data-related parameters
DATA_FOLDER = "data"
TRN_SET_SIZE = 60000
TST_SET_SIZE = 10000
N_CLASSES = 10
