import torch

__all__ = ["LogQuadraticLoss", "UncertainizedPowerLoss"]

from torch import nn


class LogQuadraticLoss(nn.Module):
    def __init__(self, alpha=10):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_hat, y):
        return torch.mean(torch.log(1 + ((self.alpha * (y_hat - y)) ** 2)))


class UncertainizedPowerLoss(nn.Module):
    def __init__(self, alpha=10, degree=2):
        super().__init__()
        self.alpha = alpha
        self.degree = degree

    def forward(self, y_hat, y, var):
        return 1 / 2 * torch.mean(
            (torch.log(var + 1e-6) +
             ((self.alpha * (y_hat - y)) ** self.degree) / (var + 1e-6))
        )
