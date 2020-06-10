import numpy as np

from config.config_loader import VALIDATION_SET_SIZE
from data_factory import get_batch

validation_set = [get_batch(batch_size=1) for _ in range(VALIDATION_SET_SIZE)]


def validate(model):
    total_error_count = 0.0

    for inputs, labels, sequence_length in validation_set:
        # Batch size for validation is 1:
        assert len(inputs) == 1
        assert len(labels) == 1
        outputs = model(inputs, sequence_length)
        assert len(outputs) == 1

        error_count = np.sum(np.abs(labels[0] - np.round(outputs[0])))
        total_error_count += error_count

    return total_error_count / VALIDATION_SET_SIZE
