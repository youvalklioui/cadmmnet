import torch
import torch.nn as nn

from utils.utils import vector_to_toeplitz, tpm
from models.soft_thresholding import SoftThresh


class UnitCellThlista(nn.Module):
    def __init__(self, v, A, beta=0.01, device="cuda"):
        super().__init__()
        self.S = SoftThresh(beta, device)
        self.v1 = torch.nn.Parameter(v.clone().detach().to(device))
        self.W2 = torch.nn.Parameter(
            A.T.conj().clone().detach().to(device).to(torch.complex128)
        )

    def forward(self, x, y):
        # Build vector that parametrizes Toeplitz matrix W1 while enforcing Hermitian symmetry
        v1 = torch.cat((torch.conj(self.v1.flip(0)), self.v1[1:]))
        # Build Toeplitz matrix W1
        W1 = vector_to_toeplitz(v1)
        # tpm(W1, x) uses an FFT to efficiently compute the product W1*x
        z = self.S(tpm(W1, x) + self.W2 @ y)
        return z


class ThLista(nn.Module):
    def __init__(self, v, A, num_layers=10, beta=0.01, device="cuda"):
        super().__init__()
        self.device = device
        self.N = v.shape[0]
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [UnitCellThlista(v, A, beta, device) for _ in range(num_layers)]
        )

    def forward(self, y):
        x = torch.zeros((self.N, y.shape[1]), dtype=y.dtype, device=self.device)
        for unit_cell in self.layers:
            x = unit_cell(x, y)
        return x

