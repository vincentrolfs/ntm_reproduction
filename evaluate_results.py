import pickle

import matplotlib.pyplot as plt
import numpy as np

from config.config_loader import VALIDATION_INTERVAL, BATCH_SIZE
from utils.base_settings import apply_base_settings

assert __name__ == '__main__'

apply_base_settings()

VALIDATION_RESULTS_PATH = 'outputs/copy_task/validation_results_1590739088_31299/validation_results.pickle'

with open(VALIDATION_RESULTS_PATH, "rb") as f:
    validation_results = pickle.load(f)

sequence_numbers = BATCH_SIZE * VALIDATION_INTERVAL * (1 + np.array(range(len(validation_results))))

print(validation_results)
plt.plot(sequence_numbers, validation_results)
plt.gca().set_xlabel("Sequence number")
plt.gca().set_ylabel("Average number of mistakes per sequence")
plt.show()
