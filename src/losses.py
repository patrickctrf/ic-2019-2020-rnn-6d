import torch

__all__ = ["LogQuadraticLoss", "UncertainizedPowerLoss", "PosAndAngleLoss"]

from torch import nn
from torch.nn import GaussianNLLLoss, CosineSimilarity, MSELoss, L1Loss


class LogQuadraticLoss(nn.Module):
    def __init__(self, alpha=10):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_hat, y):
        return torch.mean(torch.log(1 + ((self.alpha * (y_hat - y)) ** 2)))


class UncertainizedPowerLoss(nn.Module):
    def __init__(self, alpha=100, degree=2):
        super().__init__()
        self.alpha = alpha
        self.degree = degree

    def forward(self, y_hat, y, var):
        return 1 / 2 * torch.mean(torch.log(var + 1e-6)) + \
               self.alpha / 2 * torch.mean(((y_hat - y) ** self.degree) / (var + 1e-6))


def _norm_error_and_cosine_similarity(y_hat, y, p=2):
    return torch.norm(y_hat[:, :3] - y[:, :3], dim=1, p=p) - \
           torch.cosine_similarity(y_hat[:, 3:] - y[:, 3:], dim=1)


class PosAndAngleLoss(nn.Module):
    def __init__(self, alpha=100, degree=2):
        super().__init__()
        self.alpha = alpha
        self.degree = degree
        self.translational_metric = MSELoss()
        self.angular_metric = CosineSimilarity(dim=1)

    def forward(self, y_hat, y, var=0):  # , var[:, 0:3]) \
        return self.translational_metric(y_hat[:, 0:3], y[:, 0:3]) \
               - \
               torch.mean(self.angular_metric(y_hat[:, 3:], y[:, 3:]))
