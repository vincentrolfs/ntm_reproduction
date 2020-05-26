import os
import pickle
from time import time

import tensorflow as tf

from NTM import NTM
from Visualizer import Visualizer
from base_settings import apply_base_settings
from constants import *
from loss_function import get_loss_function
from optimizer import get_optimizer
from training_data import generate_training_data

apply_base_settings()
ntm = NTM()
loss_function = get_loss_function()
optimizer = get_optimizer()
training_data = generate_training_data()


def save_progress(batch_index):
    t = str(int(time()))
    model_dir = 'model_checkpoint_' + t + '_' + str(batch_index) + '/'
    optimizer_dir = 'optimizer_checkpoint_' + t + '_' + str(batch_index) + '/'

    os.mkdir(model_dir)
    ntm.save_weights(model_dir + 'model_checkpoint.ckpt')

    os.mkdir(optimizer_dir)
    with open(optimizer_dir + "optimizer_checkpoint.pickle", "wb+") as f:
        pickle.dump(optimizer, f)


def train_step(batch_index):
    inputs, labels, sequence_length = training_data[batch_index]

    with tf.GradientTape() as tape:
        outputs = ntm(inputs, sequence_length)

        # Keras's binary cross-entropy does an unexpected mean over last dimension
        # We don't want that, so we add tf.newaxis
        loss = loss_function(labels[..., tf.newaxis], outputs[..., tf.newaxis])
        loss = tf.reduce_sum(loss) / inputs.shape[0]

        gradients = tape.gradient(loss, ntm.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, MAX_GLOBAL_GRAD_NORM)

        optimizer.apply_gradients(zip(gradients, ntm.trainable_variables))

    if (batch_index + 1) % PROGRESS_SAVE_INTERVAL == 0: save_progress(ntm, optimizer, batch_index)

    return loss.numpy(), outputs


losses = []
visualizer = Visualizer()

for batch_index in range(AMOUNT_BATCHES):
    loss, outputs = train_step(batch_index)
    losses.append(loss)
    visualizer.print_progress(batch_index)
