import os
import torch

from utils.utils import randi, randu, randn, idxf, rand_freqs

PI = torch.pi

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASETS_PATH = f"{BASE_PATH}/datasets"

"""
Module for generating sensor arrays, dictionaries, and datasets.

This module includes:
- generate_array: Generate sensor placement for ULA or SLA.
- generate_dictionary: Build a dictionary of steering vectors based on a sensor array.
- single_measurement_vector: Simulate a noisy measurement from a sparse source signal.
- generate_dataset_train: Create and save a training dataset.
- generate_dataset_test: Create and save a test dataset.
"""


def generate_array(array_type='SLA', num_elements=20, aperture=50):
    """
    Generate sensor array coordinates.

    For 'SLA' (sparse linear array), randomly generate sensor positions.
    For 'ULA' (uniform linear array), use consecutive indices.

    Args:
        array_type (str): Type of array ('SLA' or 'ULA').
        num_elements (int): Number of sensor elements.
        aperture (int): Aperture (range) for SLA.

    Returns:
        torch.Tensor: Array containing sensor locations.
    """
    if array_type == 'SLA':
        # Generate and sort random sensor positions between 1 and (aperture - 1)
        array, _ = torch.sort(randi(1, aperture - 1, num_elements - 2))
        # Append fixed endpoints 0 and aperture.
        array = torch.cat((torch.tensor([0]), array, torch.tensor([aperture])))
    elif array_type == 'ULA':
        array = torch.arange(num_elements)
        print("The aperture is ignored for the ULA setting.")
    return array


def generate_dictionary(array_type='SLA', num_elements=20, aperture=50, dictionary_length=256):
    """
    Generate a dictionary for sparse representation.

    The dictionary is based on sensor array positions and a frequency grid.
    It saves the dictionary in the appropriate folder.

    Args:
        array_type (str): Array type ('SLA' or 'ULA').
        num_elements (int): Number of sensor elements.
        aperture (int): Aperture for SLA (ignored for ULA).
        dictionary_length (int): Number of frequency grid points.
    """
    # Create frequency grid from -1/2 to 1/2
    freq_grid = torch.arange(-1 / 2, 1 / 2, 1 / dictionary_length)
    array = generate_array(array_type, num_elements, aperture)
    if array_type == 'ULA':
        # For ULA, the aperture is simply num_elements - 1.
        aperture = num_elements - 1

    # Build dictionary matrix using the steering vector formula.
    A = torch.exp(1j * 2 * PI * array.unsqueeze(-1) * freq_grid.unsqueeze(0))
    A = A.to(torch.complex128)

    metadata = {
        'array_type': array_type,
        'num_elements': num_elements,
        'aperture': aperture if array_type == 'SLA' else num_elements - 1,
        'dictionary_length': dictionary_length
    }

    dictionary = {
        'metadata': metadata,
        'array': array,
        'frequency_grid': freq_grid,
        'dictionary': A
    }

    dictionary_tag = f"dictionary_{num_elements}elem_{aperture}ap_{dictionary_length}len"
    dictionary_path = f"{DATASETS_PATH}/{array_type.lower()}"
    os.makedirs(dictionary_path, exist_ok=True)
    torch.save(dictionary, f"{dictionary_path}/{dictionary_tag}.pt")

    print(f"Dictionary saved as {dictionary_tag}.pt under {dictionary_path}.")


def single_measurement_vector(array, snr, max_number_sources, min_freq_separation, frequency_grid):
    """
    Generate a single measurement vector with noise.

    Args:
        array (torch.Tensor): Sensor array positions.
        snr (list): Signal-to-noise ratio; can be a single value list or range.
        max_number_sources (int): Maximum number of active sources.
        min_freq_separation (float): Minimum separation between frequencies.
        frequency_grid (torch.Tensor): Frequency grid for sparse representation.

    Returns:
        tuple: Transposed measurement vector and sparse representation.
    """
    
    num_signals = randi(1, max_number_sources, 1)
    
    # Randomly select frequencies with a specified minimum separation.
    freqs = rand_freqs(-1 / 2, 1 / 2, min_freq_separation, num_signals)
    # Generate random amplitudes and phases.
    amps = randu((num_signals, 1), 0, 1)
    phis = randu((num_signals, 1), -PI, PI)
    # Create complex amplitudes.
    amps = amps * torch.exp(1j * phis).to(torch.complex128)

    # Build measurement vector using the steering matrix.
    y = torch.matmul(
        torch.exp(1j * 2 * PI * array.unsqueeze(-1) * freqs.unsqueeze(0)).to(torch.complex128),
        amps
    )
    sig_power = 1 / len(y) * torch.sum(torch.abs(y) ** 2)

    # Handle SNR: if a single value is provided, use it directly.
    if len(snr) == 1:
        snr = snr[0]
    else:
        snr = randu(snr[0], snr[1], 1)

    sigma = torch.sqrt(sig_power / 2) * 10 ** ((-snr) / 20)
    # Add complex Gaussian noise.
    noise = randn((len(y), 1), 0, sigma) + 1j * randn((len(y), 1), 0, sigma)
    y = y + noise

    # Build sparse representation vector.
    x = torch.zeros((len(frequency_grid), 1), dtype=torch.complex128)
    indices = idxf(freqs, frequency_grid)
    x[indices] = amps

    return y.T, x.T


def generate_dataset_train(dictionary_path, num_measurement_vectors, max_number_sources, snr=[15],
                           min_freq_separation_factor=1):
    """
    Generate and save a training dataset.

    The dataset consists of measurement vectors and their corresponding sparse vectors.
    """
    # Load array metadata, sensor array, and frequency grid.
    array_info = torch.load(dictionary_path, weights_only=True)
    array_type = array_info['metadata']['array_type'].lower()
    array = torch.load(dictionary_path, weights_only=True)['array']
    frequency_grid = torch.load(dictionary_path, weights_only=True)['frequency_grid']

    array_size = len(array)
    dictionary_length = len(frequency_grid)
    min_freq_separation = 1 / (min_freq_separation_factor * array_size)

    measurement_vectors = torch.zeros((num_measurement_vectors, array_size), dtype=torch.complex128)
    sparse_vectors = torch.zeros((num_measurement_vectors, dictionary_length), dtype=torch.complex128)

    metadata = {
        'array_type': array_type,
        'array': array,
        'num_training_vectors': num_measurement_vectors,
        'max_number_sources': max_number_sources,
        'snr': snr,
        'min_freq_separation_factor': min_freq_separation,
        'dictionary_length': dictionary_length,
        'dictionary_path': dictionary_path
    }

    print("Generating training dataset...")
    for s in range(num_measurement_vectors):
        measurement_vectors[s, :], sparse_vectors[s, :] = single_measurement_vector(
            array, snr, max_number_sources, min_freq_separation, frequency_grid
        )

    dataset = {
        'metadata': metadata,
        'data': {
            'measurement_vectors': measurement_vectors,
            'sparse_vectors': sparse_vectors
        }
    }

    snr_tag = f"{snr[0]}dbsnr" if len(snr) == 1 else f"{snr[0]}to{snr[1]}dbsnr"
    dataset_tag = f"dataset_train_{max_number_sources}tgts_{snr_tag}_{min_freq_separation_factor}fres"
    dataset_path = f"{DATASETS_PATH}/{array_type}"
    os.makedirs(dataset_path, exist_ok=True)
    torch.save(dataset, f"{dataset_path}/{dataset_tag}.pt")

    print(f"Training dataset is saved as {dataset_tag}.pt under {dataset_path}.")


def generate_dataset_test(dictionary_path, snr_values, num_vectors_per_snr, max_number_sources,
                          min_freq_separation_factor=3):
    """
    Generate and save a test dataset.

    The dataset includes measurement vectors generated for each SNR value.
    """
    # Load array metadata, sensor array, and frequency grid.
    array_info = torch.load(dictionary_path, weights_only=True)
    array_type = array_info['metadata']['array_type'].lower()
    array = torch.load(dictionary_path, weights_only=True)['array']
    frequency_grid = torch.load(dictionary_path, weights_only=True)['frequency_grid']

    array_size = len(array)
    dictionary_length = len(frequency_grid)
    min_freq_separation = 1 / (min_freq_separation_factor * array_size)
    len_dataset = num_vectors_per_snr * len(snr_values)

    measurement_vectors = torch.zeros((len_dataset, array_size), dtype=torch.complex128)
    sparse_vectors = torch.zeros((len_dataset, dictionary_length), dtype=torch.complex128)

    metadata = {
        'array_type': array_type,
        'array': array,
        'num_vectors_per_snr': num_vectors_per_snr,
        'max_number_sources': max_number_sources,
        'snr_values': snr_values,
        'min_freq_separation_factor': min_freq_separation_factor,
        'dictionary_length': dictionary_length,
        'dictionary_path': dictionary_path
    }

    print("Generating test dataset...")
    s = 0
    for snr in snr_values:
        for t in range(num_vectors_per_snr):
            measurement_vectors[s, :], sparse_vectors[s, :] = single_measurement_vector(
                array, [snr], max_number_sources, min_freq_separation, frequency_grid
            )
            s += 1

    dataset = {
        'metadata': metadata,
        'data': {
            'measurement_vectors': measurement_vectors,
            'sparse_vectors': sparse_vectors
        }
    }

    dataset_tag = (
        f"dataset_test_{max_number_sources}tgts_"
        f"{min(snr_values)}to{max(snr_values)}dbsnr_"
        f"{min_freq_separation_factor}fres"
    )
    dataset_path = f"{DATASETS_PATH}/{array_type}"
    os.makedirs(dataset_path, exist_ok=True)
    torch.save(dataset, f"{dataset_path}/{dataset_tag}.pt")

    print(f"Test dataset saved as {dataset_tag}.pt under {dataset_path}.")