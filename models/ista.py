import torch


def ista(y, A, beta=1e-2, niter=200):
    N = A.shape[1]
    mu = 1 / torch.linalg.matrix_norm(A, ord=2) ** 2
    x = torch.zeros((N, y.shape[1]), dtype=y.dtype, device=y.device)

    for _ in range(niter):
        x = (
            (torch.eye(N, device=y.device) - mu * A.T.conj() @ A) @ x
            + mu * A.T.conj() @ y
        )
        x = torch.exp(1j * x.angle()) * torch.max(
            torch.abs(x) - beta,
            torch.zeros(x.size(), device=y.device)
        )

    return x

