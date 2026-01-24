import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        features = self.X[index]
        target = self.y[index]
        
        return features, target
    
    def create_dataloader(self, batch_size: int, shuffle: bool = True):
        return DataLoader(self, batch_size=batch_size, shuffle = shuffle)
    
    @property
    def num_features(self):
        return self.X.shape[1]
