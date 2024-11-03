from transformers import Trainer

from data.partitioner import partition_data
from train.misc import cross_difference_loss
from train.train import initialize_submodels, average_model_weights


class MISTTrainer:
    def __init__(self, model, dataset, num_submodels, cross_diff_lambda=0.01, split_name="train"):
        self.model = model
        self.dataset = dataset
        self.num_submodels = num_submodels
        self.cross_diff_lambda = cross_diff_lambda
        self.submodels = initialize_submodels(model, num_submodels)
        self.partitions = partition_data(dataset, num_submodels, split_name=split_name)

    def train(self, training_args):
        # Initialize trainers for each submodel
        self.trainers = []  # Define trainers as a class attribute
        for i, submodel in enumerate(self.submodels):
            trainer = Trainer(
                model=submodel,
                args=training_args,
                train_dataset=self.partitions[i],
                compute_metrics=self.compute_metrics
            )
            self.trainers.append(trainer)

        # Train each submodel
        for trainer in self.trainers:
            trainer.train()

        # Apply cross-difference loss across submodels
        self.apply_cross_difference_loss()

        # Aggregate weights to produce the global model
        global_weights = average_model_weights(self.submodels)
        self.model.load_state_dict(global_weights)

    def apply_cross_difference_loss(self):
        # Apply cross-difference loss regularization
        outputs = [trainer.predict(self.partitions[i]).predictions for i, trainer in enumerate(self.trainers)]
        loss = cross_difference_loss(outputs)
        return loss * self.cross_diff_lambda

    def compute_metrics(self, pred):
        # Custom metric computations (optional)
        # Implement ROC-AUC, PLR, or other relevant MI metrics here if desired~
        pass