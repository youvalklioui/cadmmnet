import torch
import torch.nn as nn


class SoftThresh(nn.Module):
    def __init__(self, beta, device="cuda"):
        super().__init__()
        self.beta = torch.nn.Parameter(beta * torch.ones([1], device=device))
        self.device = device

    def forward(self, x):
        zeros = torch.zeros(x.size(), device=self.device)
        x = torch.exp(1j * x.angle()) * torch.max(
            torch.abs(x) - self.beta, zeros
        )
        return x