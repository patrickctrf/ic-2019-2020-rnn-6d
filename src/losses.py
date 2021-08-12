import torch

__all__ = ["log_quadratic"]


def log_quadratic(y_hat, y):
    return torch.mean(torch.log(1 + ((100 * (y_hat - y)) ** 2)))
