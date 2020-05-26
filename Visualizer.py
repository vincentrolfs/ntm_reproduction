from time import time

import matplotlib.pyplot as plt
import numpy as np

from constants import *


class Visualizer:
    def __init__(self, start_time=None):
        self.start_time = start_time or time()
        self.last_display_time = -1

    def print(self, *args):
        args_formatted = []

        for arg in args:
            if isinstance(arg, float):
                arg = "{:.2f}".format(arg)
            args_formatted.append(arg)

        print(*args_formatted)

    def print_progress(self, batch_index):
        if (time() - self.last_display_time) < TRAINING_PROGRESS_DISPLAY_INTERVAL: return

        self.last_display_time = time()

        global_progress = batch_index / AMOUNT_BATCHES
        ran_for = time() - self.start_time

        self.print('Global progress:', batch_index, 'steps out of', AMOUNT_BATCHES, '-', 100 * global_progress,
                   '% done')
        self.print('  Ran for', ran_for, 'seconds')
        self.print('  Expected running time', ran_for / global_progress if global_progress > 0 else '?', 'seconds')
        self.print('  Expected remaining time', ran_for / global_progress - ran_for if global_progress > 0 else '?',
                   'seconds')

    def show_errors(self, batch_index, labels, sequence_length, outputs):
        plt.figure(figsize=(15, 7))

        assert labels.shape == (BATCH_SIZE, sequence_length, NUM_BITS_PER_VECTOR)
        assert outputs.shape == (BATCH_SIZE, sequence_length, NUM_BITS_PER_VECTOR)

        errors = np.abs(labels - np.round(outputs))
        errors = np.append(errors, 2 * np.ones((BATCH_SIZE, 4, NUM_BITS_PER_VECTOR)),
                           axis=1)  # We add "fake" vectors to seperate the sequences
        errors = errors.reshape(-1, errors.shape[-1])

        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['white', '#DD7373', 'black'])

        plt.imshow(errors, aspect='auto', interpolation='none', cmap=cmap)
        plt.gca().set_title('Errors batch #' + str(batch_index + 1))

        plt.savefig(ERROR_VISUALIZATION_PATH)
