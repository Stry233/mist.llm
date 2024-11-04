import numpy as np
import logging

logger = logging.getLogger(__name__)

def partition_data(dataset, num_local_models):
    """
    Partition the dataset into disjoint subsets for each local optim.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be partitioned.
        num_local_models (int): Number of local models, determines the number of partitions.

    Returns:
        list of numpy.ndarray: Each element is an array of indices representing a partition.

    Behavior:
        Shuffles the dataset indices and splits them into `num_local_models` disjoint subsets.

    Requirements:
        - `dataset` must implement `__len__` to provide dataset size.
        - `num_local_models` should be a positive integer less than or equal to dataset size.
    """
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    partitions = np.array_split(indices, num_local_models)
    logger.info("Data partitioned into %d subsets.", num_local_models)
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
