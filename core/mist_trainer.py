import logging

import torch
from transformers import Trainer

from data.mock import MockDataset
from core.mist_callback import MISTCallback

logger = logging.getLogger(__name__)


class MISTTrainer(Trainer):
    def __init__(self, model, dataset, eval_dataset, num_local_models, T1, T2, cross_diff_weight, repartition=True, **kwargs):
        super().__init__(model=model, train_dataset=MockDataset(), eval_dataset=eval_dataset, **kwargs)
        self.dataset = dataset
        self.num_local_models = num_local_models
        self.local_models = [model.to(self.args.device) for _ in range(num_local_models)]
        self.optimizers = [torch.optim.Adam(local_model.parameters(), lr=self.args.learning_rate) for local_model in
                           self.local_models]

        # Initialize the MISTCallback with phase-specific arguments
        mist_callback = MISTCallback(
            num_local_models=num_local_models,
            T1=T1,
            T2=T2,
            cross_diff_weight=cross_diff_weight,
            repartition=repartition,
            dataset=dataset,
            model=model,
            local_models=self.local_models,
            optimizers=self.optimizers
        )

        # Add MISTCallback to the list of callbacks
        self.add_callback(mist_callback)

    def training_step(self, *args, **kwargs):
        # Return a dummy tensor with zero loss
        return torch.tensor(0.0, device=self.args.device)
