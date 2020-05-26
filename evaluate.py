from Model import Model
from data_factory import get_training_data_batch
from utils.StatusMonitor import StatusMonitor
from utils.base_settings import apply_base_settings

apply_base_settings()
ntm = Model()
monitor = StatusMonitor()
inputs, labels, sequence_length = get_training_data_batch()
outputs = ntm(inputs, sequence_length)

monitor.save_error_visualization(0, labels, outputs, sequence_length)
