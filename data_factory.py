import numpy as np

from config.config_loader import MAX_SEQUENCE_LENGTH, BATCH_SIZE, NUM_BITS_PER_VECTOR, MIN_SEQUENCE_LENGTH

snap_boolean = np.vectorize(lambda x: 1.0 if x > 0.5 else 0.0)


def get_sequence(sequence_length):
    return np.asarray([
        snap_boolean(np.append(np.random.rand(NUM_BITS_PER_VECTOR), 0))
        for _ in range(sequence_length)
    ])


def get_batch(batch_size=BATCH_SIZE):
    sequence_length = np.random.randint(low=MIN_SEQUENCE_LENGTH, high=MAX_SEQUENCE_LENGTH + 1)

    main_inputs = np.asarray([get_sequence(sequence_length) for _ in range(batch_size)]).astype(np.float32)
    end_of_sequence_marker = np.ones([batch_size, 1, NUM_BITS_PER_VECTOR + 1])
    empty_inputs = np.zeros_like(main_inputs)

    inputs = np.concatenate((main_inputs, end_of_sequence_marker, empty_inputs), axis=1).astype('float32')
    labels = main_inputs[:, :, :NUM_BITS_PER_VECTOR].astype('float32')

    return inputs, labels, sequence_length
