import torch
import torch.nn as nn

from models.soft_thresholding import SoftThresh





    
class UnitCellCAdmmNet(nn.Module):
    def __init__(self, v, beta=0.1, rho=1.0, device="cuda"):
        super().__init__()
        self.device = device
        self.S = SoftThresh(beta, device=device)
        self.v = torch.nn.Parameter(v.clone().detach().to(device))
        self.rho = torch.nn.Parameter(
            rho * torch.ones([1], device=device, dtype=torch.float64)
        )

    def forward(self, u_in, yf):
        
        # Perform inversion in eigen-domain
        w = 1 / (self.v + self.rho).unsqueeze(-1)

        u_out = torch.fft.ifft(w * torch.fft.fft(self.rho * (2 * self.S(u_in) - u_in) + yf, dim=0), dim=0)
        u_out = u_out + u_in - self.S(u_in)
        
        return u_out

    
class CAdmmNet(nn.Module):
    def __init__(self, v, A, num_layers=15, beta=0.1, rho=1.0, device="cuda"):
        super().__init__()
        self.device = device
        self.S = SoftThresh(beta, device=device)
        self.A = A.to(device)
        self.N = A.shape[1]
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [UnitCellCAdmmNet(v, beta, rho, device) for _ in range(num_layers)]
        )
        
    
    def forward(self, y):
        yf = self.A.T.conj() @ y
        u = torch.zeros((self.N, y.shape[1]), dtype=y.dtype, device=self.device)

        for unit_cell in self.layers:
            u = unit_cell(u, yf)

        return self.S(u)