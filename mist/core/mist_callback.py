import logging
from transformers import TrainerCallback, TrainerControl, TrainerState
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm  # Import tqdm for progress bars
from mist.optim.misc import local_training, cross_difference_loss, aggregate_model_states
from mist.data.partition import repartition_data

logger = logging.getLogger(__name__)


class MISTCallback(TrainerCallback):
    def __init__(self, num_local_models, T1, T2, cross_diff_weight, repartition=True, **kwargs):
        self.num_local_models = num_local_models
        self.T1 = T1
        self.T2 = T2
        self.cross_diff_weight = cross_diff_weight
        self.repartition = repartition
        self.local_models = kwargs["local_models"]
        self.optimizers = kwargs["optimizers"]
        self.dataset = kwargs["dataset"]
        self.data_partitions = repartition_data(self.dataset, self.num_local_models)

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info("Starting MIST two-phase training.")

        # Phase 1: Diverse Local Training
        local_model_states = []
        with tqdm(total=self.num_local_models, desc="Phase 1: Local Model Training", unit="model") as pbar1:
            for i, local_model in enumerate(self.local_models):
                dataloader = DataLoader(Subset(self.dataset, self.data_partitions[i]), batch_size=args.train_batch_size)
                local_state_dict = local_training(local_model, self.optimizers[i], dataloader, self.T1, args.device)
                local_model_states.append(local_state_dict)
                logger.info(f"Phase 1: Completed training for local model {i + 1}/{self.num_local_models}")
                pbar1.update(1)  # Update progress bar for each completed model

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
