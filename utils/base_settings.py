import numpy as np
import tensorflow as tf

from config.config_loader import *


def apply_base_settings():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.debugging.set_log_device_placement(True)
