import copy
from transformers import Trainer, TrainingArguments

def initialize_submodels(model, num_submodels):
    return [copy.deepcopy(model) for _ in range(num_submodels)]


def average_model_weights(models):
    # Get a list of all model state dictionaries
    state_dicts = [model.state_dict() for model in models]
    # Initialize an empty state dictionary for the global model
    global_state_dict = copy.deepcopy(state_dicts[0])

    # Average each parameter across all models
    for key in global_state_dict.keys():
        global_state_dict[key] = sum(state_dict[key] for state_dict in state_dicts) / len(models)

    return global_state_dict
