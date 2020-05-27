import os
import pickle
from time import time

from config.config_loader import *


class FileSaver:
    def save_model(self, model, batch_index):
        model_dir = self.format_path(MODEL_SAVE_PATH_PREFIX, batch_index)

        os.mkdir(model_dir)
        model.save_weights(model_dir + MODEL_SAVE_FILENAME)

    def save_optimizer(self, optimizer, batch_index):
        optimizer_dir = self.format_path(OPTIMIZER_SAVE_PATH_PREFIX, batch_index)

        os.mkdir(optimizer_dir)
        with open(optimizer_dir + OPTIMIZER_SAVE_FILENAME, "wb+") as f:
            pickle.dump(optimizer, f)

    def format_path(self, base, batch_index, append='/'):
        t = str(int(time()))
        return base + '_' + t + '_' + str(batch_index) + append


file_saver = FileSaver()
