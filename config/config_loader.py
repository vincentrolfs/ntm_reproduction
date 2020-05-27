import argparse

import json5

parser = argparse.ArgumentParser(description='Train or validate the NTM implemenation.')
parser.add_argument('config_file_path', type=str, help='path to the config file')
parser.add_argument('--load_model', metavar='MODEL_CHECKPOINT_PATH', type=str,
                    help='path to the model checkpoint. If not given, a new, untrained model is used.')
parser.add_argument('--load_optimizer', metavar='OPTIMIZER_CHECKPOINT_PATH', type=str,
                    help='path to the optimizer checkpoint. If not given, a newly initialized optimizer is used.')
args = parser.parse_args()


def load_config():
    config_path_defaults = 'config/_defaults.json5'
    config_path_given = args.config_file_path

    with open(config_path_defaults, 'r') as file_defaults, open(config_path_given, 'r') as file_given:
        config = json5.load(file_defaults)
        config.update(json5.load(file_given))

        return config


config = load_config()

MODEL_LOAD_PATH = args.load_model
OPTIMIZER_LOAD_PATH = args.load_optimizer

CONFIG_NAME = config['CONFIG_NAME']
AMOUNT_BATCHES = config['AMOUNT_BATCHES']
BATCH_SIZE = config['BATCH_SIZE']
MAX_SEQUENCE_LENGTH = config['MAX_SEQUENCE_LENGTH']
MIN_SEQUENCE_LENGTH = config['MIN_SEQUENCE_LENGTH']
NUM_BITS_PER_VECTOR = config['NUM_BITS_PER_VECTOR']
PROGRESS_SAVE_INTERVAL = config['PROGRESS_SAVE_INTERVAL']
TRAINING_PROGRESS_DISPLAY_INTERVAL = config['TRAINING_PROGRESS_DISPLAY_INTERVAL']

SEED = config['SEED']

OUTPUTS_DIR = config['OUTPUTS_DIR']
MODEL_SAVE_PATH_PREFIX = config['MODEL_SAVE_PATH_PREFIX']
MODEL_SAVE_FILENAME = config['MODEL_SAVE_FILENAME']
OPTIMIZER_SAVE_PATH_PREFIX = config['OPTIMIZER_SAVE_PATH_PREFIX']
OPTIMIZER_SAVE_FILENAME = config['OPTIMIZER_SAVE_FILENAME']

CONTROLLER_NUM_LAYERS = config['CONTROLLER_NUM_LAYERS']
CONTROLLER_NUM_UNITS_PER_LAYER = config['CONTROLLER_NUM_UNITS_PER_LAYER']

USE_RMSPROP_OPTIMIZER = config['USE_RMSPROP_OPTIMIZER']
LEARNING_RATE = config['LEARNING_RATE']
MOMENTUM = config['MOMENTUM']
DECAY = config['DECAY']
MAX_GLOBAL_GRAD_NORM = config['MAX_GLOBAL_GRAD_NORM']

ERROR_VISUALIZATION_FILENAME = config['ERROR_VISUALIZATION_FILENAME']
ERROR_VISUALIZATION_FILE_EXTENSION = config['ERROR_VISUALIZATION_FILE_EXTENSION']
