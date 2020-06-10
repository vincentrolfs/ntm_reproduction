import pickle

from utils.base_settings import apply_base_settings

apply_base_settings()

PATH = 'outputs/copy_task/validation_results_1590739088_31299/validation_results.pickle'

with open(PATH, "r+") as f:
    validation_results = pickle.load(f)

print(validation_results)
