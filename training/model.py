from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging

logger = logging.getLogger(__name__)

class IrisModel(nn.Module):
    def __init__(self,X_tensor: torch.tensor,y_tensor: torch.tensor, hidden_dim:int , dropout:int = 3, batch_size = 32):
        super(IrisModel, self).__init__()
        
        self.X = X_tensor
        self.y = y_tensor
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.batch_size = batch_size
        
        self.dataset = TensorDataset(X_tensor, y_tensor)
        
        self.net = nn.Sequential(
            nn.Linear(X_tensor.shape[1],hidden_dim),
            nn.Softmax(),
            nn.Dropout(p = dropout)
        )
        
    def forward(self, x):
        return self.net(x)
    
    def _create_dataloader(self):
        return DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True)
    
    