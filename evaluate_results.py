import pickle

import matplotlib.pyplot as plt
import numpy as np

from config.config_loader import VALIDATION_INTERVAL, BATCH_SIZE, VALIDATION_RESULTS_LOAD_PATH, LOSSES_LOAD_PATH, \
    CONFIG_NAME
from utilities.base_settings import apply_base_settings

assert __name__ == '__main__'

apply_base_settings()

assert LOSSES_LOAD_PATH is not None, "No path to the losses was given. Use the --help flag for more info. "

assert VALIDATION_RESULTS_LOAD_PATH is not None, "No path to the validation results was given. Use the --help flag " \
                                                 "for more info. "

with open(LOSSES_LOAD_PATH, "rb") as f:
    losses = pickle.load(f)

with open(VALIDATION_RESULTS_LOAD_PATH, "rb") as f:
    validation_results = pickle.load(f)

batch_indices = range(len(losses))
sequence_numbers = VALIDATION_INTERVAL * (1 + np.array(range(len(validation_results))))

print("Losses: \n", losses)
print("Validation results: \n", validation_results)

fig, ax = plt.subplots(1, 2, figsize=(15, 7))

ax[0].plot(batch_indices, losses)
ax[0].set_xlabel("Batch index")
ax[0].set_ylabel("Loss")
ax[0].set_title("Training losses")

ax[1].plot(sequence_numbers, validation_results)
ax[1].set_xlabel("Batch index")
ax[1].set_ylabel("Mistakes per sequence")
ax[1].set_title("Validation: Average number of mistakes per sequence")

plt.savefig("figures/" + CONFIG_NAME + ".pdf")

#plt.show()
