import torch
import logging
import copy
from transformers import Trainer


logger = logging.getLogger(__name__)

def local_training(model, dataset, T, training_args, tokenizer=None, eval_dataset=None, data_collator=None):
    """
    Conducts local training on a single model using Hugging Face Trainer with a specified number of epochs.

    Args:
        model (torch.nn.Module): Model to be trained locally.
        dataset (datasets.Dataset or torch.utils.data.Dataset): Dataset for training.
        T (int): Number of epochs to train.
        training_args (transformers.TrainingArguments): Training arguments for Hugging Face Trainer.
        tokenizer (transformers.PreTrainedTokenizer, optional): Tokenizer used for text inputs.
        eval_dataset (datasets.Dataset or torch.utils.data.Dataset, optional): Dataset for evaluation.
        data_collator (callable, optional): Function to collate data into batches.

    Returns:
        dict: State dictionary of the locally trained model.

    Behavior:
        Trains the model using Hugging Face Trainer for T epochs using the provided data and configuration.
    """
    # Update the number of epochs in training arguments
    training_args.num_train_epochs = T

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    # Start training
    trainer.train()

    # Return the model's state dict
    return model.state_dict()




def cross_difference_loss(model, optimizer, dataloader, reference_models, T, cross_diff_weight, device):
    """
    Optimizes model parameters to minimize cross-difference loss with reference models.

    Args:
        model (torch.nn.Module): Main model to optimize.
        optimizer (torch.optim.Optimizer): Optimizer for the main model.
        dataloader (torch.utils.data.DataLoader): DataLoader for local model data.
        reference_models (list of torch.nn.Module): Other models for calculating cross-difference loss.
        T (int): Number of gradient update steps.
        cross_diff_weight (float): Scaling weight for the cross-difference loss.
        device (str): Device identifier for model and data.

    Returns:
        dict: State dictionary of the updated model.

    Behavior:
        For each batch in `dataloader`, computes cross-difference loss with each reference model.
        Updates `model` parameters `T` times using the scaled loss.

    Requirements:
        - All models must be compatible with the batch structure from `dataloader`.
        - Reference models should be set to `.eval()` mode and detached from gradient computation.
    """
    criterion = torch.nn.L1Loss()
    model.train()
    model.to(device)

    for step in range(T):
        logger.info("Cross-difference loss optimization step %d/%d", step + 1, T)
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            main_outputs = model(**batch)
            main_model_confidence = torch.softmax(main_outputs.logits, dim=1)
            loss = 0

            for ref_model in reference_models:
                ref_model.to(device)
                with torch.no_grad():
                    ref_outputs = ref_model(**batch)
                    ref_confidence = torch.softmax(ref_outputs.logits, dim=1)
                loss += criterion(main_model_confidence, ref_confidence)

            loss = (cross_diff_weight / len(reference_models)) * loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model.state_dict()


def aggregate_model_states(model_states):
    """
    Aggregates a list of model states by averaging the weights.

    Args:
        model_states (list of dict): List of model state dictionaries to be averaged.

    Returns:
        dict: Averaged state dictionary.

    Behavior:
        Averages the parameters of each model in `model_states` across all keys.

    Requirements:
        - `model_states` should contain identical keys in each dictionary.
        - Each parameter tensor must support element-wise addition and division.
    """
    avg_state_dict = copy.deepcopy(model_states[0])
    for key in avg_state_dict.keys():
        avg_state_dict[key] = sum(state[key] for state in model_states) / len(model_states)
    logger.info("Aggregated model states.")
    return avg_state_dict
