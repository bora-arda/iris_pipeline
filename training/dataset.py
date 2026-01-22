import pandas as pd
import torch
from torch.utils.data import Dataset
import logging
from utils import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

class IrisDataset(Dataset):
    def __init__(self, data_dir: str):
        logger.info(f"Loading dataset from {data_dir}")
        if not data_dir.endswith(".csv"):
            raise ValueError("Only CSV files are supported")
        self.df = pd.read_csv(data_dir)
        
        if "target" not in self.df.columns:
            raise ValueError("Dataset must contain a 'target' column")
        
        # Separate features and target
        self.X = self.df.drop(columns=["target"]).values
        self.y = self.df["target"].values

        
        logger.info(f"Dataset loaded with {len(self.df)} samples")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        features = torch.tensor(self.X[index], dtype=torch.float32)
        target = torch.tensor(self.y[index], dtype=torch.long)
        
        return features, target
            