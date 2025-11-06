import torch
import torch.nn as nn

from models.soft_thresholding import SoftThresh


class UnitCellAdmmNet(nn.Module):
    def __init__(self, W, beta=2, rho=1, device="cuda"):
        super().__init__()
        self.S = SoftThresh(beta, device=device)
        self.W = torch.nn.Parameter(W.clone().detach().to(device))
        self.rho = torch.nn.Parameter(
            rho * torch.ones([1], device=device, dtype=torch.float64)
        )
        self.device = device

    def forward(self, z_in, v_in, yf, N):
        
        t= torch.max(self.rho, torch.tensor([0],device=self.device))
        W_tmp = torch.linalg.inv(
            self.W + t * torch.eye(N, dtype=torch.complex128, device=self.device)
        )

        x_out = W_tmp @ (t * (z_in - v_in) + yf)
        z_out = self.S(x_out + v_in)
        v_out = v_in + x_out - z_out

        return x_out, z_out, v_out


class AdmmNet(nn.Module):
    def __init__(self, W, A, num_layers=10, beta=2, rho=1, device="cuda"):
        super().__init__()
        self.device = device
        self.A = A.to(device)
        self.N = A.shape[1]
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [UnitCellAdmmNet(W, beta, rho, device) for _ in range(num_layers)]
        )

    def forward(self, y):
        x = torch.zeros((self.N, y.shape[1]), dtype=y.dtype, device=self.device)
        z = torch.zeros((self.N, y.shape[1]), dtype=y.dtype, device=self.device)
        v = torch.zeros((self.N, y.shape[1]), dtype=y.dtype, device=self.device)

        yf = self.A.T.conj() @ y

        for unit_cell in self.layers:
            x, z, v = unit_cell(z, v, yf, self.N)

        return x