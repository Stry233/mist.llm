import torch
from torch.utils.data import Dataset

class MockDataset(Dataset):
    def __len__(self):
        return 1  # Just a placeholder to make Hugging Face Trainer happy.

    def __getitem__(self, idx):
        # Return a dummy data sample.
        return {"input_ids": torch.zeros(1, dtype=torch.long), "labels": torch.tensor(0)}
