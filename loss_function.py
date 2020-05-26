import tensorflow as tf
from tensorflow.python import keras


def get_loss_function():
    bce_loss_function = keras.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.NONE)

    def loss_function(inputs, labels, outputs):
        # Keras's binary cross-entropy does an unexpected mean over last dimension
        # We don't want that, so we add tf.newaxis
        loss = bce_loss_function(labels[..., tf.newaxis], outputs[..., tf.newaxis])
        loss = tf.reduce_sum(loss) / inputs.shape[0]

        return loss

    return loss_function
