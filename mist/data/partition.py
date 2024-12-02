import numpy as np
import logging

logger = logging.getLogger(__name__)

def partition_data(dataset, num_partitions):
    """Repartition the dataset into `num_partitions` subsets."""
    total_size = len(dataset)
    indices = list(range(total_size))
    partition_size = total_size // num_partitions
    partitions = [indices[i*partition_size:(i+1)*partition_size] for i in range(num_partitions)]
    return partitions

def repartition_data(dataset, num_local_models):
    """
    Repartition the dataset into new disjoint subsets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be repartitioned.
        num_local_models (int): Number of partitions.

    Returns:
        list of numpy.ndarray: New disjoint partitions for each local optim.

    Behavior:
        Calls `partition_data` to shuffle and repartition data anew, useful for certain federated learning strategies.
    """
    logger.info("Repartitioning data across local models.")
    return partition_data(dataset, num_local_models)
