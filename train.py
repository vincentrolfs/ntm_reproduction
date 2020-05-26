import tensorflow as tf

from FileSaver import file_saver
from Model import Model
from Visualizer import Visualizer
from base_settings import apply_base_settings
from constants import *
from loss_function import get_loss_function
from optimizer import get_optimizer
from training_data import generate_training_data

apply_base_settings()
model = Model()
loss_function = get_loss_function()
optimizer = get_optimizer()
training_data = generate_training_data()

def train_step(inputs, labels, sequence_length):
    with tf.GradientTape() as tape:
        outputs = model(inputs, sequence_length)

        # Keras's binary cross-entropy does an unexpected mean over last dimension
        # We don't want that, so we add tf.newaxis
        loss = loss_function(labels[..., tf.newaxis], outputs[..., tf.newaxis])
        loss = tf.reduce_sum(loss) / inputs.shape[0]

        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, MAX_GLOBAL_GRAD_NORM)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss.numpy(), outputs


losses = []
visualizer = Visualizer()

for batch_index in range(AMOUNT_BATCHES):
    inputs, labels, sequence_length = training_data[batch_index]
    loss, outputs = train_step(inputs, labels, sequence_length)
    losses.append(loss)
    visualizer.print_progress(batch_index)

    if (batch_index + 1) % PROGRESS_SAVE_INTERVAL == 0:
        file_saver.save_model(model, batch_index)
        file_saver.save_optimizer(optimizer, batch_index)
        visualizer.save_error_visualization(batch_index, labels, outputs, sequence_length)
