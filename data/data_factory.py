import numpy as np

from config.config_loader import MAX_SEQUENCE_LENGTH, BATCH_SIZE, NUM_BITS_PER_VECTOR, MIN_SEQUENCE_LENGTH, SORT_LABELS, \
    REPEAT_LABELS, MIN_LABEL_REPETITIONS, MAX_LABEL_REPETITIONS, ASSOCIATIVE_RECALL_MODE, RECALL_ITEM_SIZE

if ASSOCIATIVE_RECALL_MODE:
    assert MAX_SEQUENCE_LENGTH % RECALL_ITEM_SIZE == 0, "MAX_SEQUENCE_LENGTH must be divisible by RECALL_ITEM_SIZE"
    assert MIN_SEQUENCE_LENGTH % RECALL_ITEM_SIZE == 0, "MIN_SEQUENCE_LENGTH must be divisible by RECALL_ITEM_SIZE"
    assert MIN_SEQUENCE_LENGTH // RECALL_ITEM_SIZE >= 2, "MIN_SEQUENCE_LENGTH must fit at least two items"

snap_boolean = np.vectorize(lambda x: 1.0 if x > 0.5 else 0.0)
delimiter_flags = (0, 0) if REPEAT_LABELS else (0,)


def get_sequence(sequence_length):
    return np.asarray([
        snap_boolean(np.append(np.random.rand(NUM_BITS_PER_VECTOR), delimiter_flags))
        for _ in range(sequence_length)
    ])


def sort_lexicographically(array):
    return array[np.lexsort(np.rot90(array))]


def repeat(array, num_repetitions):
    return np.tile(array, (1, num_repetitions, 1))


def get_associative_recall_batch(batch_size):
    amount_of_items = np.random.randint(low=MIN_SEQUENCE_LENGTH // RECALL_ITEM_SIZE,
                                        high=MAX_SEQUENCE_LENGTH // RECALL_ITEM_SIZE + 1)
    sequence_length = RECALL_ITEM_SIZE * amount_of_items

    main_inputs = np.asarray([get_sequence(sequence_length) for _ in range(batch_size)]).astype(np.float32)
    delimiter_vectors = np.ones([batch_size, 1, NUM_BITS_PER_VECTOR + len(delimiter_flags)])

    query_items_start_index = RECALL_ITEM_SIZE * np.random.randint(low=0, high=amount_of_items - 1,
                                                                   size=(batch_size,))
    query_items = np.array([
        main_inputs[
            i,
            query_items_start_index[i]: query_items_start_index[i] + RECALL_ITEM_SIZE,
            :
        ]
        for i in range(batch_size)
    ])

    empty_inputs = np.zeros_like(query_items)
    labels = np.array([
        main_inputs[
            i,
            query_items_start_index[i] + RECALL_ITEM_SIZE: query_items_start_index[i] + 2 * RECALL_ITEM_SIZE,
            :NUM_BITS_PER_VECTOR
        ]
        for i in range(batch_size)
    ])

    inputs = np.concatenate(
        (main_inputs, delimiter_vectors, query_items, delimiter_vectors, empty_inputs), axis=1
    ).astype('float32')

    return inputs, labels


def get_batch(batch_size=BATCH_SIZE):
    if ASSOCIATIVE_RECALL_MODE:
        return get_associative_recall_batch(batch_size)

    sequence_length = np.random.randint(low=MIN_SEQUENCE_LENGTH, high=MAX_SEQUENCE_LENGTH + 1)

    main_inputs = np.asarray([get_sequence(sequence_length) for _ in range(batch_size)]).astype(np.float32)
    end_of_sequence_marker = np.ones([batch_size, 1, NUM_BITS_PER_VECTOR + len(delimiter_flags)])
    empty_inputs = np.zeros_like(main_inputs)
    labels = main_inputs[:, :, :NUM_BITS_PER_VECTOR].astype('float32')

    if SORT_LABELS:
        labels = np.array([sort_lexicographically(batch) for batch in labels])

    if REPEAT_LABELS:
        num_repetitions = np.random.randint(low=MIN_LABEL_REPETITIONS, high=MAX_LABEL_REPETITIONS + 1)
        end_of_sequence_marker[:, :, -1] = num_repetitions / MAX_LABEL_REPETITIONS
        empty_inputs = repeat(empty_inputs, num_repetitions)
        labels = repeat(labels, num_repetitions)

    inputs = np.concatenate((main_inputs, end_of_sequence_marker, empty_inputs), axis=1).astype('float32')

    return inputs, labels
