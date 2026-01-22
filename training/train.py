from torch import nn
import torch
from training.model import IrisModel
from training.dataset import IrisDataset
import logging

logger = logging.getLogger(__name__)

class IrisTrainer:
    def __init__(self):
        pass
    
    def prepare_training(dataset_model: IrisDataset, nn_model: IrisModel):
        
        model_dataset = dataset_model("data/processed/train.csv")
        
        
    
    def train(num_epoch: int,):
        pass