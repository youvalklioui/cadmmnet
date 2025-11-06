import torch
import torch.nn as nn

from utils.utils import vector_to_toeplitz, tpm
from models.soft_thresholding import SoftThresh

"""
Implementation of the Toeplitz-LISTA model as described in:
"Structured LISTA for Multidimensional Harmonic Retrieval"
Source: https://arxiv.org/pdf/2102.11663
"""


class UnitCellTlista(nn.Module):
    def __init__(self, v, A, beta=0.01, device="cuda"):
        super().__init__()
        self.S = SoftThresh(beta, device)
        self.v1 = torch.nn.Parameter(v.clone().detach().to(device))
        self.W2 = torch.nn.Parameter(
            A.T.conj().clone().detach().to(device).to(torch.complex128)
        )

    def forward(self, x, y):
        # Build Toeplitz matrix parametrized by v1.
        W1 = vector_to_toeplitz(self.v1)
        # tpm(W1, x) uses an FFT to efficiently compute the product W1*x.
        z = self.S(tpm(W1, x) + self.W2 @ y)
        return z


class TLista(nn.Module):
    def __init__(self, v, A, num_layers=10, beta=0.01, device="cuda"):
        super().__init__()
        self.device = device
        self.N = (v.shape[0] + 1) // 2
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [UnitCellTlista(v, A, beta, device) for _ in range(num_layers)]
        )

    def forward(self, y):
        x = torch.zeros((self.N, y.shape[1]), dtype=y.dtype, device=self.device)
        for unit_cell in self.layers:
            x = unit_cell(x, y)
        return x

