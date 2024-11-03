from datasets import DatasetDict


def partition_data(dataset, num_partitions, split_name="train"):
    # Check if the dataset is a DatasetDict with the specified split
    if isinstance(dataset, DatasetDict):
        if split_name in dataset:
            data = dataset[split_name]
        else:
            raise ValueError(
                f"Dataset does not contain a '{split_name}' split. Available splits are: {list(dataset.keys())}")
    else:
        # Handle case where dataset is already a single split
        data = dataset

    # Partition the data into `num_partitions` subsets
    partition_size = int(len(data) / num_partitions)
    partitions = [data.select(range(i * partition_size, (i + 1) * partition_size)) for i in range(num_partitions - 1)]
    partitions.append(data.select(range((num_partitions - 1) * partition_size, len(data))))

    return partitions
