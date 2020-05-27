import tensorflow as tf

from Model import Model
from config.config_loader import *
from data_factory import generate_training_data
from training.loss_function import get_loss_function
from training.optimizer import get_optimizer
from utils.StatusMonitor import StatusMonitor
from utils.base_settings import apply_base_settings
from utils.file_saver import file_saver

apply_base_settings()
model = Model()
loss_function = get_loss_function()
optimizer = get_optimizer()
training_data = generate_training_data()


def train_step(inputs, labels, sequence_length):
    with tf.GradientTape() as tape:
        outputs = model(inputs, sequence_length)
        loss = loss_function(inputs, labels, outputs)

        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, MAX_GLOBAL_GRAD_NORM)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss.numpy(), outputs


losses = []
monitor = StatusMonitor()

for batch_index in range(AMOUNT_BATCHES):
    inputs, labels, sequence_length = training_data[batch_index]
    loss, outputs = train_step(inputs, labels, sequence_length)
    losses.append(loss)
    monitor.print_progress(batch_index)

    if (batch_index + 1) % PROGRESS_SAVE_INTERVAL == 0:
        file_saver.save_model(model, batch_index)
        file_saver.save_optimizer(optimizer, batch_index)
        monitor.save_error_visualization(batch_index, labels, outputs, sequence_length)
