from Model import Model
from Visualizer import Visualizer
from base_settings import apply_base_settings
from training_data import get_training_data_batch

apply_base_settings()
ntm = Model()
visualizer = Visualizer()
inputs, labels, sequence_length = get_training_data_batch()
outputs = ntm(inputs, sequence_length)

visualizer.save_error_visualization(0, labels, sequence_length, outputs)
