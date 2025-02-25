import torch


def admm(y, A, rho=1, beta=1e-1, niter=20):
    N = A.shape[1]
    H = torch.linalg.inv(
        A.conj().T @ A + rho * torch.eye(N, dtype=torch.complex128, device=y.device)
    )
    x = torch.zeros((N, y.shape[1]), dtype=y.dtype, device=y.device)
    z = torch.zeros((N, y.shape[1]), dtype=y.dtype, device=y.device)
    v = torch.zeros((N, y.shape[1]), dtype=y.dtype, device=y.device)

    for _ in range(niter):
        x = H @ (rho * (z - v) + A.conj().T @ y)
        z = x + v
        z = torch.exp(1j * z.angle()) * torch.max(
            torch.abs(z) - beta, torch.zeros(z.size(), device=y.device)
        )
        v = v + x - z

    return x