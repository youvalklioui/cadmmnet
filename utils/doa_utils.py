import torch



def fbss(y, L):

    '''
    Build the sample covariance matrix using the forward-backward averaging method.

    Inputs:
    y: measurement vector (needs to be from a ULA)
    L: number of elements in each subarray

    Returns:
    R: spatially smoothed sample covariance matrix of shape (L, L)

    '''
    M = y.shape[0]  # Equivalent to max(size(y))

    # Number of subarrays
    K = M - L + 1

    # Initialize the matrices
    Ravg = torch.zeros((L, L), dtype=torch.complex64)
    Ravg2 = torch.zeros((L, L), dtype=torch.complex64)
    
    y=y.unsqueeze(-1)

    # Loop to compute Ravg and Ravg2
    for k in range(K):
        # Compute Ravg for each k
        segment = y[k:k+L]
        Ravg += segment @ segment.conj().T

        # Reverse and conjugate the segment for Ravg2
        segment_conj = torch.flip(torch.conj(y[M-k-L:M-k]), dims=[0])
        Ravg2 += segment_conj @ segment_conj.conj().T

    # Average the results
    R = (1 / K) * 0.5 * (Ravg + Ravg2)
    
    return R




def music_spectrum_v2(y, K, L, N):

    '''
    Computes the MUSIC pseudo-spectrum at N  uniformly spaced normalized frequency points between [-1/2, 1/2-1/N] given a single-snapshot measurement vector y from a ULA an, estimate of the number of sources K, and the number of elements in each subarray L to be used for the forward-backward spatial smoothing.
    '''

    R = fbss(y,L)

    # Determine the size of the covariance matrix
    M = R.shape[0]

    # Eigen-decomposition: Since R is Hermitian, use eigh for efficiency
    eigval, eigvec = torch.linalg.eigh(R)

    # Sort eigenvalues by descending magnitude and sort eigenvectors accordingly
    eigval_abs = torch.abs(eigval)
    sorted_indices = torch.argsort(eigval_abs, descending=True)
    eigvec_sorted = eigvec[:, sorted_indices]

    # Noise subspace: eigenvectors corresponding to the smallest M-K eigenvalues
    Un = eigvec_sorted[:, K:M]
    
    # Generate the frequency grid
    freqs = torch.arange(-0.5, 0.5, 1/N, device=R.device, dtype=torch.float32).unsqueeze(0)

    array = torch.arange(0, M, device=R.device, dtype=torch.float32).unsqueeze(-1)

    A = torch.exp(1j * 2 * torch.pi * array * freqs).conj().T
    
    spectrum = 1/torch.norm(A@Un, dim=1)

    spectrum_normalized = spectrum/torch.max(spectrum)
    
    return spectrum_normalized

