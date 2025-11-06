import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

from utils.initialization_utils import initialize_model, load_state
from utils.utils import model_id, nmse, generate_kernel, conv

utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(utils_dir)

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATES_PATH = f"{BASE_PATH}/models/states"
LOSSES_PATH = f"{BASE_PATH}/outputs/losses"


class MeasurementVectorsDataset(Dataset):
    """
    PyTorch Dataset subclass for the measurement vectors and sparse vectors.
    """

    def __init__(self, dataset, device='cuda'):
        self.measurement_vectors = dataset['measurement_vectors'].to(device)
        self.sparse_vectors = dataset['sparse_vectors'].to(device)

    def __len__(self):
        return self.measurement_vectors.shape[0]

    def __getitem__(self, idx):
        return self.measurement_vectors[idx], self.sparse_vectors[idx]


def training_setup(dataset_train_path, num_training_samples=1e5, batch_size=2048,
                   device='cuda'):
    """
    Splits the dataset into training and validation sets and creates the corresponding data loaders.

    Inputs:
    - dataset_train_path: Path to the dataset.
    - num_training_samples: Number of samples to use for training (the rest will be used for validation).
    - batch_size: Batch size for the data loaders.
    - device: Device to use for training.
    """
    dataset = torch.load(dataset_train_path, weights_only=True)['data']
    
    # Convert num_training_samples to int if it's a float
    num_training_samples = int(num_training_samples)

    training_data = {key: value[:num_training_samples] for key, value in dataset.items()}
    validation_data = {key: value[num_training_samples:] for key, value in dataset.items()}

    dataset_training = MeasurementVectorsDataset(training_data, device)
    dataset_validation = MeasurementVectorsDataset(validation_data, device)

    data_loader_training = DataLoader(dataset_training, batch_size=batch_size, shuffle=True)
    data_loader_validation = DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)

    return data_loader_training, data_loader_validation


def save_state(network, optimizer, array_type, training_loss, validation_loss):
    """
    Saves the model state, optimizer state, and the losses to the corresponding directories.

    Inputs:
    - network: An instance of the model to save.
    - optimizer: An instance of the optimizer used for training.
    - array_type: The type of the array used in the dataset.
    - training_loss: The training loss tensor.
    - validation_loss: The validation loss tensor.
    """
    # Get model name and create a timestamped model tag.
    model_name, model_tag = model_id(network)
    num_layers = network.num_layers

    state = {
        'model_state': network.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'model_tag': model_tag
    }

    loss = {
        'training_loss': training_loss,
        'validation_loss': validation_loss
    }

    state_path = f"{STATES_PATH}/{array_type}/{model_name}/{num_layers}l"
    loss_path = f"{LOSSES_PATH}/{array_type}/{model_name}/{num_layers}l"

    os.makedirs(state_path, exist_ok=True)
    os.makedirs(loss_path, exist_ok=True)

    torch.save(state, f"{state_path}/{model_tag}.pt")
    torch.save(loss, f"{loss_path}/{model_tag}_loss.pt")

    return state_path, loss_path, model_tag



def train_model(model, dataset_train_path, num_layers=30, epochs=30, lr=1e-4, batch_size=2048,                  
                num_training_samples=100000, model_path=None, load_latest_state=False, device='cuda'):
    
    """
    Trains a model on the training dataset for a given number of epochs.

    Inputs:
    - model: The model to train. The model can be "LISTA", "ADMM-Net", "TLISTA", "THLISTA", "CADMM-Net", "CHADMM-Net".
    - dataset_train_path: Path to the training dataset.
    - num_layers: Number of layers in the model.
    - epochs: Number of epochs to train.
    - lr: Learning rate for the optimizer.
    - batch_size: Batch size for the data loaders.
    - num_training_samples: Number of samples to use for training (the rest will be used for validation).
    - model_path: Optional path to the model state to resume training from.
    - load_latest_state: If True, the latest model state will be loaded to resume training.
    - device: Device to use for training.
    """
    data_loader_training, data_loader_validation = training_setup(
        dataset_train_path, num_training_samples, batch_size, device)

    metadata = torch.load(dataset_train_path, weights_only=True)['metadata']
    array_type = metadata['array_type'].lower()
    dictionary_path = metadata['dictionary_path']

    network = initialize_model(model, dictionary_path, num_layers, device)
    optimizer = torch.optim.Adam(network.parameters(), lr)

    KERNEL_SCALE = 0.5
    KERNEL_LENGTH = 7
    kernel = generate_kernel(KERNEL_LENGTH, KERNEL_SCALE, device)

    if load_latest_state or model_path:
        load_state(network, optimizer, array_type, load_latest_state, model_path, return_tag=False)

    training_loss_batch = torch.zeros(len(data_loader_training))
    validation_loss_batch = torch.zeros(len(data_loader_validation))

    training_loss = torch.zeros(epochs)
    validation_loss = torch.zeros(epochs)

 

    print("Training ...")

    for epoch in range(epochs):

        network.train()

        for batch_idx, (Y, X) in enumerate(data_loader_training):
            X, Y = X.T, Y.T

            X_est = network(Y)

            loss = nmse(conv(X_est,kernel), conv(X,kernel))

            network.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss_batch[batch_idx] = loss.item()

        network.eval()

        for batch_idx, (Yv, Xv) in enumerate(data_loader_validation):
            Xv, Yv = Xv.T, Yv.T

            with torch.no_grad():
                X_estv = network(Yv)

            loss = loss = nmse(conv(X_estv,kernel), conv(Xv,kernel))
            validation_loss_batch[batch_idx] = loss.item()

        training_loss[epoch] = torch.mean(training_loss_batch)
        validation_loss[epoch] = torch.mean(validation_loss_batch)

        print(
            f"Epoch {epoch+1}/{epochs} - Training Loss: {10*torch.log10(training_loss[epoch])} dB - "
            f"Validation Loss: {10*torch.log10(validation_loss[epoch])} dB"
        )

    state_path, loss_path, model_tag = save_state(network, optimizer, array_type,
                                                  training_loss, validation_loss)

    print(
        f"Model state and losses saved under '{state_path}' and '{loss_path}', respectively.\n"
        f"Model tag: {model_tag}"
    )




