import numpy as np

from utils.constants import *

snap_boolean = np.vectorize(lambda x: 1.0 if x > 0.5 else 0.0)


def generate_sequence(sequence_length):
    return np.asarray([
        snap_boolean(np.append(np.random.rand(NUM_BITS_PER_VECTOR), 0))
        for _ in range(sequence_length)
    ])


def get_training_data_batch():
    sequence_length = MAX_SEQUENCE_LENGTH  # np.random.randint(low=MIN_SEQUENCE_LENGTH, high=MAX_SEQ_LEN+1)

    main_inputs = np.asarray([generate_sequence(sequence_length) for _ in range(BATCH_SIZE)]).astype(np.float32)
    end_of_sequence_marker = np.ones([BATCH_SIZE, 1, NUM_BITS_PER_VECTOR + 1])
    empty_inputs = np.zeros_like(main_inputs)

    inputs = np.concatenate((main_inputs, end_of_sequence_marker, empty_inputs), axis=1).astype('float32')
    labels = main_inputs[:, :, :NUM_BITS_PER_VECTOR].astype('float32')

    return inputs, labels, sequence_length


def generate_training_data():
    return [get_training_data_batch() for _ in range(AMOUNT_BATCHES)]
