import torch
import copy
import logging
from transformers import Trainer
from torch.utils.data import DataLoader, Subset
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MISTTrainer(Trainer):
    def __init__(self, model, dataset, num_local_models, T1, T2, cross_diff_weight, epochs, repartition=True, **kwargs):
        super().__init__(model, **kwargs)
        self.dataset = dataset
        self.num_local_models = num_local_models
        self.T1 = T1  # Number of gradient updates for Phase 1
        self.T2 = T2  # Number of gradient updates for Phase 2
        self.cross_diff_weight = cross_diff_weight  # Cross-difference weight
        self.epochs = epochs
        self.repartition = repartition  # Flag for repartitioning at each epoch
        self.local_models = [copy.deepcopy(model).to(self.args.device) for _ in range(num_local_models)]
        self.optimizers = [torch.optim.Adam(local_model.parameters(), lr=self.args.learning_rate) for local_model in self.local_models]
        self.data_partitions = self._partition_data()  # Initial partition

    def _partition_data(self):
        # Partition dataset into `num_local_models` disjoint subsets
        indices = np.arange(len(self.dataset))
        np.random.shuffle(indices)
        partitions = np.array_split(indices, self.num_local_models)
        return partitions

    def repartition_data(self):
        # Repartition data if required, maintaining disjoint subsets
        if self.repartition:
            logger.info("Repartitioning data across local models.")
            self.data_partitions = self._partition_data()

    def _local_training(self, model, optimizer, dataloader, T):
        model.train()
        model.to(self.args.device)  # Ensure the model is on the correct device

        for step in range(T):
            logger.info(f"Local training step {step + 1}/{T}")
            for batch in dataloader:
                # Move batch data to the device
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        return model.state_dict()

    def _cross_difference_loss(self, model, optimizer, dataloader, reference_models, T):
        criterion = torch.nn.L1Loss()  # Using L1 as per the MIST paper
        model.train()
        model.to(self.args.device)  # Ensure the model is on the correct device

        for step in range(T):
            logger.info(f"Cross-difference loss optimization step {step + 1}/{T}")
            for batch in dataloader:
                # Move batch data to the device
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                main_outputs = model(**batch)
                main_model_confidence = torch.softmax(main_outputs.logits, dim=1)
                loss = 0

                # Calculate cross-difference loss with reference models
                for ref_model in reference_models:
                    ref_model.to(self.args.device)
                    with torch.no_grad():  # Reference models do not need gradients
                        ref_outputs = ref_model(**batch)
                        ref_confidence = torch.softmax(ref_outputs.logits, dim=1)
                    loss += criterion(main_model_confidence, ref_confidence)

                # Scale the loss by cross_diff_weight and number of reference models
                loss = (self.cross_diff_weight / len(reference_models)) * loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        return model.state_dict()

    def train(self):
        logger.info("Starting MIST training...")
        aggregated_state_dict = self.model.state_dict()

        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")

            # Repartition dataset if the flag is set
            self.repartition_data()

            # Phase 1: Diverse Model Exploration for each local model
            local_model_states = []
            for i, local_model in enumerate(self.local_models):
                logger.info(f"Phase 1: Training local model {i + 1}/{self.num_local_models}")
                dataloader = DataLoader(Subset(self.dataset, self.data_partitions[i]), batch_size=self.args.train_batch_size)
                local_state_dict = self._local_training(local_model, self.optimizers[i], dataloader, self.T1)
                local_model_states.append(local_state_dict)

            # Phase 2: Minimize Cross-Difference Loss
            updated_model_states = []
            for i, local_model in enumerate(self.local_models):
                logger.info(f"Phase 2: Cross-difference loss optimization for local model {i + 1}/{self.num_local_models}")
                dataloader = DataLoader(Subset(self.dataset, self.data_partitions[i]), batch_size=self.args.train_batch_size)
                # Set all other models as reference for cross-difference loss
                reference_models = [self.local_models[j] for j in range(self.num_local_models) if j != i]
                updated_state_dict = self._cross_difference_loss(local_model, self.optimizers[i], dataloader, reference_models, self.T2)
                updated_model_states.append(updated_state_dict)

            # Aggregation: Average all local models to form the global model
            logger.info("Aggregating model parameters across local models.")
            aggregated_state_dict = self._aggregate(updated_model_states)
            self.model.load_state_dict(aggregated_state_dict)

        logger.info("MIST training completed.")
        return self.model

    def _aggregate(self, model_states):
        # Average the weights of all local models
        avg_state_dict = copy.deepcopy(model_states[0])
        for key in avg_state_dict.keys():
            avg_state_dict[key] = sum(state[key] for state in model_states) / len(model_states)
        return avg_state_dict
