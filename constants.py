SEED = 0

OPTIMIZER_LOAD_PATH = 'optimizer_checkpoint_1590197499_29999/optimizer_checkpoint.pickle'
MODEL_LOAD_PATH = 'model_checkpoint_1590197499_29999/model_checkpoint.ckpt'

OPTIMIZER_SAVE_PATH_PREFIX = 'optimizer_checkpoint'
OPTIMIZER_SAVE_FILENAME = 'optimizer_checkpoint.pickle'
MODEL_SAVE_PATH_PREFIX = 'model_checkpoint'
MODEL_SAVE_FILENAME = 'model_checkpoint.ckpt'

AMOUNT_BATCHES = 10  # How many batches should we use?
BATCH_SIZE = 32  # How many sequences should one batch contain?
MAX_SEQUENCE_LENGTH = 20  # How many vectors should a sequence contain at maximum?
MIN_SEQUENCE_LENGTH = 2  # How many vectors should a sequence contain at minimum?
NUM_BITS_PER_VECTOR = 8  # How many bits should a vector contain?

CONTROLLER_NUM_LAYERS = 1
CONTROLLER_NUM_UNITS_PER_LAYER = 100

USE_RMSPROP_OPTIMIZER = False  # If False, we will use Adam optimizer
LEARNING_RATE = 0.001
MOMENTUM = 0.9  # Only for RMSPROP
DECAY = 0.9  # Only for RMSPROP
MAX_GLOBAL_GRAD_NORM = 50

TRAINING_PROGRESS_DISPLAY_INTERVAL = 5  # In seconds
PROGRESS_SAVE_INTERVAL = 1  # In train steps

ERROR_VISUALIZATION_FILENAME = 'errors'
ERROR_VISUALIZATION_FILE_EXTENSION = '.pdf'
