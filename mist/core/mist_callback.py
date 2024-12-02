import logging
import os

from transformers import TrainerCallback, TrainerControl, TrainerState
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm  # Import tqdm for progress bars
from mist.optim.misc import local_training, cross_difference_loss, aggregate_model_states
from mist.data.partition import repartition_data

logger = logging.getLogger(__name__)


class MISTCallback(TrainerCallback):
    def __init__(self, num_local_models, T1, T2, cross_diff_weight, repartition=True, **kwargs):
        """
        Args:
            num_local_models (int): Number of local models to train.
            T1 (int): Number of epochs for phase 1 local training.
            T2 (int): Number of epochs for phase 2 local training.
            cross_diff_weight (float): Weight for cross-difference loss in phase 2.
            repartition (bool): Whether to repartition the dataset for local models.
            kwargs: Additional arguments:
                - `local_models`: List of local model instances.
                - `dataset`: The full training dataset to be partitioned.
                - `eval_dataset`: The evaluation dataset.
                - `training_args`: Shared TrainingArguments instance.
                - `tokenizer`: Tokenizer for text data.
                - `data_collator`: Callable for collating batches.
        """
        self.num_local_models = num_local_models
        self.T1 = T1
        self.T2 = T2
        self.cross_diff_weight = cross_diff_weight
        self.repartition = repartition
        self.local_models = kwargs["local_models"]
        self.eval_dataset = kwargs["eval_dataset"]
        self.optimizers = kwargs["optimizers"]
        self.dataset = kwargs["dataset"]
        self.training_args = kwargs["args"]
        self.tokenizer = kwargs.get("tokenizer", None)
        self.data_collator = kwargs.get("data_collator", None)
        self.data_partitions = repartition_data(self.dataset, self.num_local_models)


    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info("Starting MIST two-phase training.")

        # Phase 1: Diverse Local Training
        local_model_states = []
        with tqdm(total=self.num_local_models, desc="Phase 1: Local Model Training", unit="model") as pbar1:
            for i, local_model in enumerate(self.local_models):
                # Create a unique output directory for each local model
                original_output_dir = self.training_args.output_dir
                original_log_dir = self.training_args.logging_dir
                self.training_args.output_dir = os.path.join(original_output_dir, f"local_model_{i}")

                # Modify logging and reporting for each local model
                if self.training_args.logging_dir is not None:
                    self.training_args.logging_dir = os.path.join(original_log_dir, f"local_model_{i}")
                    self.training_args.run_name = self.training_args.output_dir
                    # self.training_args.report_to = []  # Disable reporting for local models if necessary

                # Create the dataset partition
                local_dataset = Subset(self.dataset, self.data_partitions[i])

                # Call local_training
                local_state_dict = local_training(
                    model=local_model,
                    dataset=local_dataset,
                    T=self.T1,
                    training_args=self.training_args,
                    tokenizer=self.tokenizer,
                    eval_dataset=self.eval_dataset,
                    data_collator=self.data_collator
                )
                local_model_states.append(local_state_dict)

                # Restore the original output directory
                self.training_args.output_dir = original_output_dir
                self.training_args.original_log_dir = original_log_dir
                logger.info(f"Phase 1: Completed training for local model {i + 1}/{self.num_local_models}")
                pbar1.update(1)

        # Store the local model states for further processing
        self.local_model_states = local_model_states

        # Phase 2: Cross-Difference Loss Minimization
        updated_model_states = []
        with tqdm(total=self.num_local_models, desc="Phase 2: Cross-Difference Loss Optimization", unit="model") as pbar2:
            for i, local_model in enumerate(self.local_models):
                dataloader = DataLoader(Subset(self.dataset, self.data_partitions[i]), batch_size=args.train_batch_size)
                reference_models = [self.local_models[j] for j in range(self.num_local_models) if j != i]
                updated_state_dict = cross_difference_loss(
                    local_model, self.optimizers[i], dataloader, reference_models, self.T2, self.cross_diff_weight,
                    args.device
                )
                updated_model_states.append(updated_state_dict)
                logger.info(f"Phase 2: Completed cross-difference optimization for local model {i + 1}/{self.num_local_models}")
                pbar2.update(1)  # Update progress bar for each completed model

        # Aggregation of updated model states
        global_state_dict = aggregate_model_states(updated_model_states)
        kwargs["model"].load_state_dict(global_state_dict)
        logger.info("MIST training epoch completed and global model updated.")
