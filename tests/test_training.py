import torch
from core.model import IrisModel


def test_model_forward_pass():
    model = IrisModel(input_dim=4, hidden_dim=16, num_classes=3, dropout=0.1)
    x = torch.randn(2, 4)
    y = model(x)
    assert y.shape == (2, 3)
