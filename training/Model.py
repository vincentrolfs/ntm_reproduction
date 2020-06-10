import sys

import tensorflow as tf
from tensorflow.python import keras

from config.config_loader import CONTROLLER_NUM_LAYERS, CONTROLLER_NUM_UNITS_PER_LAYER, NUM_BITS_PER_VECTOR, \
    MODEL_LOAD_PATH

# This dirty hack is needed in order to import NTMCell correctly
# If the next line is missing, we cannot import successfully, because
# ntm/ntm.py tries to import from ntm/utils.py, which Python does not find
sys.path.append("ntm/")
from ntm import NTMCell
sys.path.pop()

class Model(keras.Model):
    def __init__(self):
        super().__init__()

        ntm_cell = NTMCell(
            controller_layers=CONTROLLER_NUM_LAYERS,
            controller_units=CONTROLLER_NUM_UNITS_PER_LAYER,
            memory_size=128,
            memory_vector_dim=20,
            read_head_num=1,
            write_head_num=1,
            addressing_mode='content_and_location',
            shift_range=1,
            output_dim=NUM_BITS_PER_VECTOR,
            clip_value=20,
            init_mode='constant'
        )

        self.rnn = keras.layers.RNN(
            cell=ntm_cell,
            return_sequences=True,
            return_state=False,
            stateful=False,
            unroll=True
        )

        if MODEL_LOAD_PATH is not None:
            self.load_weights(MODEL_LOAD_PATH)

    def __call__(self, inputs, sequence_length):
        output_sequence = self.rnn(inputs)
        output_logits = output_sequence[:, sequence_length + 1:, :]
        outputs = tf.sigmoid(output_logits)

        return outputs
