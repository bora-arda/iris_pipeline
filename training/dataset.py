import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

class IrisDataset(Dataset):
    def __init__(self, data_dir: str):
        logger.info(f"Loading dataset from {data_dir}")
        if not data_dir.endswith(".csv"):
            raise ValueError("Only CSV files are supported")
        
        df = pd.read_csv(data_dir)
        
        if "target" not in df.columns:
            raise ValueError("Dataset must contain a 'target' column")
        
        # Separate features and target
        self.X = torch.tensor(
            df.drop(columns=["target"]).values, 
            dtype=torch.float32)
        
        self.y = torch.tensor(
            df["target"].values,
            dtype=torch.long)

        
        logger.info(f"Dataset loaded with {len(df)} samples")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        features = self.X[index]
        target = self.y[index]
        
        return features, target
    
    
    @property
    def num_features(self):
        return self.X.shape[1]

    def create_dataloader(self, batch_size: int):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True
        )
