import tensorflow as tf
from tensorflow.python import keras

def get_loss_function():
    return keras.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.NONE)
