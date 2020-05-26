TRAIN_STEPS = 30000 # How many batches should we use?
BATCH_SIZE = 32 # How many sequences should one batch contain?
MAX_SEQUENCE_LENGTH = 20 # How many vectors should a sequence contain at maximum?
MIN_SEQUENCE_LENGTH = 2  # How many vectors should a sequence contain at minimum?
NUM_BITS_PER_VECTOR = 8 # How many bits should a vector contain?

CONTROLLER_NUM_LAYERS = 1
CONTROLLER_NUM_UNITS_PER_LAYER = 100

MAX_GLOBAL_GRAD_NORM = 50

USE_RMSPROP_OPTIMIZER = False # If False, we will use Adam optimizer
LEARNING_RATE = 0.001
MOMENTUM = 0.9 # Only for RMSPROP
DECAY = 0.9 # Only for RMSPROP

TRAINING_PROGRESS_DISPLAY_INTERVAL = 60 # In seconds
PROGRESS_SAVE_INTERVAL = 3000 # In train steps

OPTIMIZER_LOAD_PATH = None
NTM_LOAD_PATH = None

SEED = 0