from Model import Model
from data_factory import get_training_data_batch
from utils.Visualizer import Visualizer
from utils.base_settings import apply_base_settings

apply_base_settings()
ntm = Model()
visualizer = Visualizer()
inputs, labels, sequence_length = get_training_data_batch()
outputs = ntm(inputs, sequence_length)

visualizer.save_error_visualization(0, labels, outputs, sequence_length)
