import os
import sys

import torch

from utils.utils import toeplitz_to_vector, circulant_hermitian_to_vector
from models import Lista, AdmmNet, CAdmmNet, ChAdmmNet, ThAdmmNet, ThLista, TLista


utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(utils_dir)

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATES_PATH = f"{BASE_PATH}/models/states"


def initialize_model(model, dictionary_path, num_layers, device="cuda"):
    """
    Initializes a model with the specified architecture and number of layers.
    """
    # load dictionary
    A = torch.load(dictionary_path, weights_only=True)["dictionary"]

    # compute the Gram matrix
    B = torch.matmul(A.T.conj(), A)

    # get the length of the dictionary
    N = A.shape[1]

    # compute the step-size for ISTA-based networks
    mu = 1 / torch.linalg.matrix_norm(A, ord=2) ** 2

    # Initialize the learnable ADMM penalty parameter (this initial value will work for most cases)
    rho = 1

    # Initialize the thresholding level for the soft-thresholding operator
    beta = 1e-1

    if model == "LISTA":
        network = Lista(torch.eye(N) - mu * B, mu * A, num_layers, beta, device)
    elif model == "ADMM-Net":
        network = AdmmNet(B, A, num_layers, beta, rho, device)
    elif model == "CADMM-Net":
        v0 = B[:, 0]
        v0 = torch.fft.fft(v0)
        network = CAdmmNet(v0, A, num_layers, beta, rho, device)
    elif model == "CHADMM-Net":
        v0 = circulant_hermitian_to_vector(B)
        network = ChAdmmNet(v0, A, num_layers, beta, rho, device)
    elif model == "THADMM-Net":
        v0 = B[:, 0]
        network = ThAdmmNet(v0, A, num_layers, beta, rho, device)
    elif model == "THLISTA":
        v0 = (torch.eye(N) - mu * B)[:, 0]
        network = ThLista(v0, mu * A, num_layers, beta, device)
    elif model == "TLISTA":
        v0 = toeplitz_to_vector(torch.eye(N) - mu * B)
        network = TLista(v0, mu * A, num_layers, beta, device)
    else:
        raise ValueError(
            f"Model '{model}' not recognized. Please choose from 'LISTA', 'ADMM-Net', "
            f"'CADMM-Net','CHADMM-Net', 'THADMM-Net', 'THLISTA', 'TLISTA'."
        )

    return network


def load_state(
    network, optimizer=None, array_type="sla", load_latest_state=True, model_path=None, return_tag=False
):
    """
    Loads the state of a model either for training or metric evaluation from a file. 
    If model_path is not specified, the latest model state will be loaded from the states directory.
    Set load_latest_state to False to specify a model state path.
    """
    if load_latest_state:
        if model_path is not None:
            raise ValueError(
                "Please set load_latest_state to False when specifying a model state path."
            )
        else:
            model_name = network.__class__.__name__.replace("-", "").lower()
            num_layers = network.num_layers
            state_path = f"{STATES_PATH}/{array_type}/{model_name}/{num_layers}l"
            model_id_prefix = f"{model_name}_{num_layers}l"

            matching_states = [
                state.name
                for state in os.scandir(state_path)
                if state.is_file() and state.name.startswith(model_id_prefix)
            ]

            if not matching_states:
                raise FileNotFoundError(
                    f"No model state files were found for '{model_name.upper()}' with {num_layers} "
                    f"layers. Please run again with load_latest_state set to False."
                )

            latest_model_state = max(
                matching_states,
                key=lambda f: os.path.getmtime(os.path.join(state_path, f)),
            )

            model_path = f"{state_path}/{latest_model_state}"
            print(f"Loading latest model state: '{model_path}' ...")
    else:
        print(f"Loading model state from: '{model_path}' ...")


    model_state = torch.load(model_path, weights_only=True)["model_state"]
    network.load_state_dict(model_state)

    if optimizer is not None:
        print("Loading optimizer state ...")
        optimizer_state = torch.load(model_path, weights_only=True)["optimizer_state"]
        optimizer.load_state_dict(optimizer_state)

    if return_tag:

        return torch.load(model_path, weights_only=True)["model_tag"]





