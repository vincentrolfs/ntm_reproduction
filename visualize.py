from NTM import NTM
from Visualizer import Visualizer
from training_data import get_training_data_batch

ntm = NTM()
visualizer = Visualizer()
inputs, labels, sequence_length = get_training_data_batch()
outputs = ntm(inputs, sequence_length)

visualizer.show_errors(labels, sequence_length, outputs)
