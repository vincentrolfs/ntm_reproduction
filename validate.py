import numpy as np

from Model import Model
from config.config_loader import VALIDATION_SET_SIZE
from data_factory import get_batch
from utils.base_settings import apply_base_settings

validation_set = [get_batch(batch_size=1) for _ in range(VALIDATION_SET_SIZE)]

def validate(model):
    total_error_count = 0.0

    for inputs, labels, sequence_length in validation_set:
        assert len(inputs) == 1
        assert len(labels) == 1
        outputs = model(inputs, sequence_length)
        assert len(outputs) == 1

        error_count = np.sum(np.abs(labels[0] - np.round(outputs[0])))
        total_error_count += error_count

    return total_error_count / VALIDATION_SET_SIZE


if __name__ == '__main__':
    apply_base_settings()
    model = Model()
    print(validate(model))
