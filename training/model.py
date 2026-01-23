from torch import nn
import logging

logger = logging.getLogger(__name__)

class IrisModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim:int , num_classes: int, dropout:float = 0.3):
        super(IrisModel, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)
    
    