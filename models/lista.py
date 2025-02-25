import torch
import torch.nn as nn

from models.soft_thresholding import SoftThresh


class UnitCellLista(nn.Module):
    def __init__(self, W1, W2, beta=0.01, device="cuda"):
        super().__init__()
        self.S = SoftThresh(beta, device)
        self.W1 = torch.nn.Parameter(W1.clone().detach().to(device))
        self.W2 = torch.nn.Parameter(W2.T.conj().clone().detach().to(device))

    def forward(self, x_in, y):
        x_out = self.S(self.W1 @ x_in + self.W2 @ y)
        return x_out


class Lista(nn.Module):
    def __init__(self, W1, W2, num_layers=10, beta=0.01, device="cuda"):
        super().__init__()
        self.device = device
        self.N = W1.shape[0]
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [UnitCellLista(W1, W2, beta, device) for _ in range(num_layers)]
        )

    def forward(self, y):
        x = torch.zeros((self.N, y.shape[1]), dtype=y.dtype, device=self.device)
        for unit_cell in self.layers:
            x = unit_cell(x, y)
        return x

