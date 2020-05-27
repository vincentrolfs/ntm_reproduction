import os
import pickle
from pathlib import Path
from time import time

from config.config_loader import MODEL_SAVE_PATH_PREFIX, MODEL_SAVE_FILENAME, OPTIMIZER_SAVE_PATH_PREFIX, \
    OPTIMIZER_SAVE_FILENAME, CONFIG_NAME, OUTPUTS_DIR, ERROR_VISUALIZATION_FILENAME, ERROR_VISUALIZATION_PATH_PREFIX, \
    LOSSES_PATH_PREFIX, LOSSES_FILENAME, VALIDATION_RESULTS_PATH_PREFIX, VALIDATION_RESULTS_FILENAME


class FileSaver:
    def __init__(self):
        Path(OUTPUTS_DIR + CONFIG_NAME).mkdir(parents=True, exist_ok=True)

    def save_model(self, model, batch_index):
        model_dir = self._format_path(MODEL_SAVE_PATH_PREFIX, batch_index)

        os.mkdir(model_dir)
        model.save_weights(model_dir + MODEL_SAVE_FILENAME)

    def save_optimizer(self, optimizer, batch_index):
        self._save_pickle(optimizer, batch_index, OPTIMIZER_SAVE_PATH_PREFIX, OPTIMIZER_SAVE_FILENAME)

    def save_losses(self, losses, batch_index):
        self._save_pickle(losses, batch_index, LOSSES_PATH_PREFIX, LOSSES_FILENAME)

    def save_validation_results(self, validation_results, batch_index):
        self._save_pickle(validation_results, batch_index, VALIDATION_RESULTS_PATH_PREFIX, VALIDATION_RESULTS_FILENAME)

    def save_error_visualization(self, fig, batch_index):
        error_visualization_dir = self._format_path(ERROR_VISUALIZATION_PATH_PREFIX, batch_index)

        os.mkdir(error_visualization_dir)
        fig.savefig(error_visualization_dir + ERROR_VISUALIZATION_FILENAME)

    def _save_pickle(self, object, batch_index, path_prefix, filename):
        dir = self._format_path(path_prefix, batch_index)

        os.mkdir(dir)
        with open(dir + filename, "wb+") as f:
            pickle.dump(object, f)

    def _format_path(self, base, batch_index, append='/'):
        t = str(int(time()))
        return OUTPUTS_DIR + CONFIG_NAME + '/' + base + '_' + t + '_' + str(batch_index) + append


file_saver = FileSaver()
