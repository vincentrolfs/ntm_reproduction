import tensorflow as tf
from tensorflow.python import keras

from constants import *
from ntm.ntm import NTMCell


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

        if NTM_LOAD_PATH is not None:
            self.load_weights(NTM_LOAD_PATH)

    def __call__(self, inputs, sequence_length):
        output_sequence = self.rnn(inputs)
        output_logits = output_sequence[:, sequence_length + 1:, :]
        outputs = tf.sigmoid(output_logits)

        return outputs
