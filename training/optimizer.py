import pickle

import tensorflow as tf

from utils.constants import *


def get_optimizer():
    if OPTIMIZER_LOAD_PATH is None:
        if USE_RMSPROP_OPTIMIZER:
            return tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM,
                                             decay=DECAY)
        else:
            return tf.optimizers.Adam(lr=LEARNING_RATE)
    else:
        with open(OPTIMIZER_LOAD_PATH, "rb") as f:
            return pickle.load(f)
