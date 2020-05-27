import os
import pickle
from time import time

import matplotlib.pyplot as plt

from config.config_loader import MODEL_SAVE_PATH_PREFIX, MODEL_SAVE_FILENAME, OPTIMIZER_SAVE_PATH_PREFIX, \
    OPTIMIZER_SAVE_FILENAME, CONFIG_NAME, OUTPUTS_DIR, ERROR_VISUALIZATION_FILENAME, ERROR_VISUALIZATION_FILE_EXTENSION


class FileSaver:
    def __init__(self):
        os.mkdir(OUTPUTS_DIR)
        os.mkdir(OUTPUTS_DIR + CONFIG_NAME)

    def save_model(self, model, batch_index):
        model_dir = self._format_path(MODEL_SAVE_PATH_PREFIX, batch_index)

        os.mkdir(model_dir)
        model.save_weights(model_dir + MODEL_SAVE_FILENAME)

    def save_optimizer(self, optimizer, batch_index):
        optimizer_dir = self._format_path(OPTIMIZER_SAVE_PATH_PREFIX, batch_index)

        os.mkdir(optimizer_dir)
        with open(optimizer_dir + OPTIMIZER_SAVE_FILENAME, "wb+") as f:
            pickle.dump(optimizer, f)

    def save_error_visualization(self, batch_index):
        path = self._format_path(ERROR_VISUALIZATION_FILENAME, batch_index, append=ERROR_VISUALIZATION_FILE_EXTENSION)
        plt.savefig(path)

    def _format_path(self, base, batch_index, append='/'):
        t = str(int(time()))
        return OUTPUTS_DIR + CONFIG_NAME + '/' + base + '_' + t + '_' + str(batch_index) + append


file_saver = FileSaver()
