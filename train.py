import tensorflow as tf

from config.config_loader import MAX_GLOBAL_GRAD_NORM, AMOUNT_BATCHES, PROGRESS_SAVE_INTERVAL, VALIDATION_INTERVAL
from data.data_factory import get_batch
from training.Model import Model
from training.loss_function import get_loss_function
from training.optimizer import get_optimizer
from training.validate import validate
from utilities.StatusMonitor import StatusMonitor
from utilities.base_settings import apply_base_settings
from utilities.file_saver import file_saver

assert __name__ == '__main__'

apply_base_settings()
model = Model()
loss_function = get_loss_function()
optimizer = get_optimizer()


def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        outputs = model(inputs, labels.shape[1])
        loss = loss_function(inputs, labels, outputs)

        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, MAX_GLOBAL_GRAD_NORM)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss.numpy(), outputs


losses = []
validation_results = []
monitor = StatusMonitor()

for batch_index in range(AMOUNT_BATCHES):
    inputs, labels = get_batch()
    loss, outputs = train_step(inputs, labels)
    losses.append(loss)

    if (batch_index + 1) % VALIDATION_INTERVAL == 0:
        print("VALIDATING...")
        validation_results.append(validate(model))
        print("VALIDATED.")

    if (batch_index + 1) % PROGRESS_SAVE_INTERVAL == 0:
        file_saver.save_model(model, batch_index)
        file_saver.save_optimizer(optimizer, batch_index)
        file_saver.save_losses(losses, batch_index)
        file_saver.save_validation_results(validation_results, batch_index)

    monitor.print_progress(batch_index, validation_results)
