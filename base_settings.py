import numpy as np
import tensorflow as tf

from constants import *


def apply_base_settings():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.debugging.set_log_device_placement(True)
