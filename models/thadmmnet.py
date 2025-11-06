import torch
import torch.nn as nn

from utils.utils import vector_to_toeplitz
from models.soft_thresholding import SoftThresh


class UnitCellThadmmNet(nn.Module):
    def __init__(self, v, beta=0.1, rho=1.0, device="cuda"):
        super().__init__()
        self.device = device
        self.S = SoftThresh(beta, device=device)

        # v parametrizes the Toeplitz-Hermitian matrix T
        self.v = torch.nn.Parameter(v.clone().detach().to(device))

        self.rho = torch.nn.Parameter(
            rho * torch.ones([1], device=device, dtype=torch.float64)
        )

    def forward(self, z_in, v_in, yf, N):
        # Build vector that parametrizes T
        v_tmp = torch.cat((torch.conj(self.v.flip(0)), self.v[1:]))

        # Build Toeplitz matrix T
        T = vector_to_toeplitz(v_tmp)

        I = torch.eye(N, dtype=torch.complex128, device=self.device)

        # Compute the scalling factor t
        min = torch.tensor([0], device=T.device)
        t = torch.max(self.rho, min) + torch.max(-torch.linalg.eigvalsh(T)[0], min)
       
        # Enforce PSD constraint and compute the inverse of the resulting Toeplitz matrix
        W_tmp = torch.linalg.inv( T + t *I)

        x_out = W_tmp @ (t* (z_in - v_in) + yf)
        z_out = self.S(x_out + v_in)
        v_out = v_in + x_out - z_out

        return x_out, z_out, v_out


class ThAdmmNet(nn.Module):
    def __init__(self, v, A, num_layers=15, beta=0.1, rho=1.0, device="cuda"):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.N = A.shape[1]
        self.layers = nn.ModuleList(
            [UnitCellThadmmNet(v, beta, rho, device) for _ in range(num_layers)]
        )
        self.A = A.to(device)

    def forward(self, y):
        yf = self.A.T.conj() @ y
        # B= self.A.T.conj() @ self.A

        x = torch.zeros((self.N, y.shape[1]), dtype=y.dtype, device=self.device)
        z = torch.zeros((self.N, y.shape[1]), dtype=y.dtype, device=self.device)
        v = torch.zeros((self.N, y.shape[1]), dtype=y.dtype, device=self.device)

        for unit_cell in self.layers:
            x, z, v = unit_cell(z, v, yf, self.N)

        return x

