import torch

__all__ = ["LogQuadraticLoss"]

from torch import nn


class LogQuadraticLoss(nn.Module):
    def __init__(self, alpha=10):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_hat, y):
        return torch.mean(torch.log(1 + ((self.alpha * (y_hat - y)) ** 2)))
