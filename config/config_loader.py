import argparse

import json5

parser = argparse.ArgumentParser(description='Train or validate the NTM implemenation.')
parser.add_argument('config_file_path', type=str, help='path to the config file')
parser.add_argument('--load_model', metavar='MODEL_CHECKPOINT_PATH', type=str,
                    help='path to the model checkpoint. If not given, a new, untrained model is used.')
parser.add_argument('--load_optimizer', metavar='OPTIMIZER_CHECKPOINT_PATH', type=str,
                    help='path to the optimizer checkpoint. If not given, a newly initialized optimizer is used.')
parser.add_argument('--load_validation_results', metavar='VALIDATION_RESULTS_PATH', type=str,
                    help='path to the validation results when evaluating results. This parameter is ignored when '
                         'training.')
parser.add_argument('--load_losses', metavar='LOSSES_PATH', type=str,
                    help='path to the training losses when evaluating results. This parameter is ignored when '
                         'training.')
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
VALIDATION_RESULTS_LOAD_PATH = args.load_validation_results
LOSSES_LOAD_PATH = args.load_losses

CONFIG_NAME = config['CONFIG_NAME']
AMOUNT_BATCHES = config['AMOUNT_BATCHES']
BATCH_SIZE = config['BATCH_SIZE']
MAX_SEQUENCE_LENGTH = config['MAX_SEQUENCE_LENGTH']
MIN_SEQUENCE_LENGTH = config['MIN_SEQUENCE_LENGTH']
NUM_BITS_PER_VECTOR = config['NUM_BITS_PER_VECTOR']
VALIDATION_SET_SIZE = config['VALIDATION_SET_SIZE']
VALIDATION_INTERVAL = config['VALIDATION_INTERVAL']
PROGRESS_SAVE_INTERVAL = config['PROGRESS_SAVE_INTERVAL']
TRAINING_PROGRESS_DISPLAY_INTERVAL = config['TRAINING_PROGRESS_DISPLAY_INTERVAL']

SEED = config['SEED']

OUTPUTS_DIR = config['OUTPUTS_DIR']
MODEL_SAVE_PATH_PREFIX = config['MODEL_SAVE_PATH_PREFIX']
MODEL_SAVE_FILENAME = config['MODEL_SAVE_FILENAME']
OPTIMIZER_SAVE_PATH_PREFIX = config['OPTIMIZER_SAVE_PATH_PREFIX']
OPTIMIZER_SAVE_FILENAME = config['OPTIMIZER_SAVE_FILENAME']
LOSSES_PATH_PREFIX = config['LOSSES_PATH_PREFIX']
LOSSES_FILENAME = config['LOSSES_FILENAME']
VALIDATION_RESULTS_PATH_PREFIX = config['VALIDATION_RESULTS_PATH_PREFIX']
VALIDATION_RESULTS_FILENAME = config['VALIDATION_RESULTS_FILENAME']
ERROR_VISUALIZATION_PATH_PREFIX = config['ERROR_VISUALIZATION_PATH_PREFIX']
ERROR_VISUALIZATION_FILENAME = config['ERROR_VISUALIZATION_FILENAME']

CONTROLLER_NUM_LAYERS = config['CONTROLLER_NUM_LAYERS']
CONTROLLER_NUM_UNITS_PER_LAYER = config['CONTROLLER_NUM_UNITS_PER_LAYER']

USE_RMSPROP_OPTIMIZER = config['USE_RMSPROP_OPTIMIZER']
LEARNING_RATE = config['LEARNING_RATE']
MOMENTUM = config['MOMENTUM']
DECAY = config['DECAY']
MAX_GLOBAL_GRAD_NORM = config['MAX_GLOBAL_GRAD_NORM']
