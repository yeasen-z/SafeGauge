import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, X, y, device="cuda"):
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
