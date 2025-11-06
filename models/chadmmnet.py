import torch
import torch.nn as nn

from models.soft_thresholding import SoftThresh






class UnitCellChAdmmNet(nn.Module):
    def __init__(self, v, beta=0.1, rho=1.0, device="cuda"):
        super().__init__()
        self.device = device
        self.S = SoftThresh(beta, device=device)
        self.v = torch.nn.Parameter(v.clone().detach().to(device))
        self.rho = torch.nn.Parameter(
            rho * torch.ones([1], device=device, dtype=torch.float64)
        )

    def forward(self,u_in,yf,idx,N):
        # Initialize weight vector
        w = torch.zeros(N,dtype=torch.complex128,device=self.device)

        # Build weight vector from the vector self.v, of size floor(N/2) + 1, that parametrizes the Hermtian Circulant
        w[idx[0]] = self.v
        w[idx[1]] = self.v[idx[2]].flip(0).conj()

        # Perform the inversion in the eigendomain
        w = 1 / (torch.fft.fft(w,dim=0) + self.rho).unsqueeze(-1)

        u_out = torch.fft.ifft(w * torch.fft.fft(self.rho * (2 * self.S(u_in) - u_in) + yf, dim=0), dim=0)
        u_out = u_out + u_in - self.S(u_in)

        return u_out






class ChAdmmNet(nn.Module):
    def __init__(self, v, A, num_layers=15, beta=0.1, rho=1.0, device="cuda"):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.N = A.shape[1]
        self.imax = (self.N // 2) + 1
        self.S = SoftThresh(beta, device=device)
        self.A = A.to(device)

        self.layers = nn.ModuleList(
            [UnitCellChAdmmNet(v, beta, rho, device) for _ in range(num_layers)]
        )

        self.idx=( torch.arange(self.imax), 
                   torch.arange(self.imax,self.N),
                   torch.arange(1,self.N-self.imax+1))
        
        

    def forward(self,y):
        yf = self.A.T.conj() @ y
        u = torch.zeros((self.N, y.shape[1]), dtype=y.dtype, device=self.device)

        for unit_cell in self.layers:
            u = unit_cell(u, yf, self.idx, self.N)

        return self.S(u)