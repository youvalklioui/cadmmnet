import os
import sys
from datetime import datetime

import torch

from utils.initialization_utils import initialize_model, load_state
from utils.utils import f2angle, k_largest_peaks, db
from models import ista, admm

utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(utils_dir)

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
METRICS_PATH = f"{BASE_PATH}/outputs/metrics"
SPECTRUMS_PATH = f"{BASE_PATH}/outputs/spectrums"





def detect(spectrum, ground_truth, bin_threshold=2, amp_threshold=0.4,
           metric='detection_rate', return_degs=True):
    """
    Computes the detection rate or MSE of the estimated spectrum compared to the ground truth spectrum using
    an amplitude and bin threshold.

    The detection rate is the fraction of peaks in the ground truth spectrum that are detected in the estimated spectrum.
    A peak in the ground truth spectrum is successfully detected if there is a peak in the estimated spectrum that is within
    `bin_threshold` bins from it and its amplitude is at least `amp_threshold` times the amplitude of the corresponding
    ground truth peak.

    The MSE is the mean squared error between the location of each peak in the ground truth spectrum and its
    corresponding detected peak in the estimated spectrum.

    Input: estimated spectrum (tensor), ground truth spectrum (tensor), bin threshold (int),
           amplitude threshold (float), metric (str), return_degs (str)
    Output: detection rate (float) or RMSE (float)

    return_degs: if True, returns the RMSE in degrees, otherwise returns the RMSE in normalized frequency units.
    """
    N = len(spectrum)

    # Get the k largest peaks in the estimated spectrum
    spec_pk_supp = k_largest_peaks(torch.abs(spectrum), N // 2)
    spec_pk_amp = torch.abs(spectrum[spec_pk_supp])

    # Get all the peaks in the ground truth spectrum
    gt_pk_supp = torch.nonzero(ground_truth)
    gt_pk_amp = torch.abs(ground_truth[gt_pk_supp])

    # Initialize detection and error tensors of length equal to the number of peaks in the ground truth spectrum
    detections = torch.zeros_like(gt_pk_supp, dtype=torch.float64)
    errors = torch.zeros_like(gt_pk_supp, dtype=torch.float64)

    def remove_elements(A, B):
        # Create a mask where elements in A that are not in B are True
        mask = ~torch.isin(A, B)
        return A[mask]

    # Stores the set of candidate peaks that have already been ascribed to a ground truth peak
    assigned = torch.tensor([])

    for k_idx in range(len(gt_pk_supp)):
        # Find the indices of the peaks in the estimated spectrum that are within bin_threshold bins of the kth ground truth peak
        detected_indices = torch.where(torch.abs(spec_pk_supp - gt_pk_supp[k_idx]) <= bin_threshold)[0]

        # Remove indices that have already been assigned to a ground truth peak
        detected_indices = remove_elements(detected_indices, assigned)

        # If no peaks are detected, continue to the next ground truth peak
        if detected_indices.numel() == 0:
            continue

        # Sort detected indices based on the distance to the ground truth peak
        sorted_indices = torch.argsort(torch.abs(spec_pk_supp[detected_indices] - gt_pk_supp[k_idx]))
        detected_indices = detected_indices[sorted_indices]

        if metric == 'detection_rate':
            # Check if any peak meets the amplitude threshold
            if torch.any(spec_pk_amp[detected_indices] > amp_threshold * gt_pk_amp[k_idx]):
                detections[k_idx] = 1

        elif metric == 'rmse':
            # Loop over the detected indices and use the first peak that meets the amplitude threshold
            for i in detected_indices:
                if spec_pk_amp[i] >= amp_threshold * gt_pk_amp[k_idx]:
                    detections[k_idx] = 1
                    if return_degs == True:
                        errors[k_idx] = (f2angle(spec_pk_supp[i] / N) -
                                           f2angle(gt_pk_supp[k_idx] / N)) ** 2
                    else:
                        errors[k_idx] = (spec_pk_supp[i] / N - gt_pk_supp[k_idx] / N) ** 2 
                    break

        # Add the detected indices to the set of assigned peaks
        assigned = torch.cat((assigned, detected_indices))

    if metric == 'detection_rate':
        return torch.count_nonzero(detections) / len(detections)
    elif metric == 'rmse':
        if torch.sum(detections) != 0:
            return torch.sum(errors) / torch.sum(detections)
        else:
            return float('nan')


def average_metric(results, num_vectors_per_snr=1e3, metric='detection_rate'):
    """
    Returns the average metric over a batch of size 'num_vectors_per_snr' for each SNR value in the test dataset.
    'results' is a tensor of length 'num_vectors_per_snr * num_snr_values' and is the output of the "detect" function
    applied to all vectors in the test dataset.
    """
    num_snr_values = results.shape[0] // num_vectors_per_snr
    average = torch.zeros(num_snr_values, dtype=torch.float64)

    for n in range(num_snr_values):
        batch_snr = results[int(n * num_vectors_per_snr):int((n + 1) * num_vectors_per_snr)]

        if metric == 'rmse':
            batch_snr = batch_snr[~torch.isnan(batch_snr)]
            average[n] = torch.sqrt(torch.sum(batch_snr) / len(batch_snr))

        elif metric in ['detection_rate']:
            average[n] = torch.sum(batch_snr) / len(batch_snr)

        elif metric == 'nmse':
            average[n] = 1 / num_vectors_per_snr * torch.sum(batch_snr)

    return average



def evaluate_model(model, dataset_test_path, num_layers, model_path=None,
                   load_latest_state=False, metric='detection_rate',
                   bin_threshold=2, amp_threshold=0.4, return_degs='True', device='cpu'):
    """
    Evaluates the performance of a model on a test dataset and saves the results in the outputs/metrics and outputs/spectrums directories.

    Input:
        model: str, the name of the model to evaluate. Options are 'ISTA', 'ADMM', 'LISTA', 'ADMM-Net', 'CADMM-Net','CHADMM-Net', 
        'THLISTA', 'TLISTA'. 
        dataset_test_path: str, the path to the test dataset. This can be generated with the dataset_generator function in the dataset folder.
        num_layers: int, the number of layers in the model.
        model_path: str, the path to the model state to load. If None, the latest available state for the model will be loaded.
        metric: str, the metric to compute. Options are 'detection_rate', 'rmse', 'nmse'.
        bin_threshold: int, the bin threshold for peak detection.
        amp_threshold: float, the amplitude threshold for peak detection.
        return_degs: str, if True, returns the RMSE in degrees, otherwise returns the RMSE in normalized frequency units.
        device: str, the device to run the model on. Options are 'cpu' and 'cuda'.
    """
    test_data = torch.load(dataset_test_path, weights_only=True)
    ground_truth = test_data['data']['sparse_vectors'].to(device)
    measurement_vectors = test_data['data']['measurement_vectors'].T.to(device)

    array_type = test_data['metadata']['array_type'].lower()
    snr_values = test_data['metadata']['snr_values']
    num_vectors_per_snr = test_data['metadata']['num_vectors_per_snr']
    dictionary_path = test_data['metadata']['dictionary_path']

    num_test_vectors = ground_truth.shape[0]

    # Initialize the model, load its state, and compute the estimated spectrums.
    if model in ['ISTA', 'ADMM']:
        A = torch.load(dictionary_path, weights_only=True)['dictionary'].to(device)
        model_tag = f"{model.lower()}_{num_layers}iter"

        if model == 'ISTA':
            spectrums = ista(measurement_vectors, A, niter=num_layers).T.detach().cpu()
        elif model == 'ADMM':
            spectrums = admm(measurement_vectors, A, niter=num_layers).T.detach().cpu()

    elif model in ['LISTA', 'ADMM-Net', 'CHADMM-Net', 'CADMM-Net', 'THADMM-Net', 'THLISTA', 'TLISTA']:
        if model_path or load_latest_state:
            network = initialize_model(model, dictionary_path, num_layers, 'cpu')
            model_tag = load_state(network, None, array_type=array_type,
                                   load_latest_state=load_latest_state,
                                   model_path=model_path, return_tag=True)
            spectrums = network(measurement_vectors).T.detach().cpu()
        else:
            raise ValueError(f"""Please either provide a model state path for {model} or set load_latest_state to True to load
the latest available state for this model.""")

    # Compute the metric for each test vector in the test dataset.
    results = torch.zeros(num_test_vectors, dtype=torch.float64)
    print("Computing metric ...")

    if metric in ['detection_rate', 'rmse']:
        for s in range(num_test_vectors):
            results[s] = detect(
                torch.abs(spectrums[s, :]),
                torch.abs(ground_truth[s, :]),
                bin_threshold,
                amp_threshold,
                metric,
                return_degs
            )
            
    
    elif metric == 'nmse':
        results = torch.norm(spectrums - ground_truth, dim=1) ** 2 / torch.norm(ground_truth, dim=1) ** 2    

    average = average_metric(results, num_vectors_per_snr, metric)

    meta_data = {
        'model': model,
        'array_type': array_type,
        'num_layers': num_layers,
        'model_tag': model_tag,
        'num_test_vectors': num_test_vectors
    }

    metric_dict = {
        'meta_data': meta_data,
        'metric': metric,
        'average': average,
        'snr_values': snr_values,
    }

    if metric in ['rmse', 'detection_rate']:
        metric_dict['amp_threshold'] = amp_threshold
        metric_dict['bin_threshold'] = bin_threshold

    spectrums_dict = {
        'meta_data': meta_data,
        'spectrums': spectrums,
        'ground_truth': ground_truth
    }

    model_name = model.replace("-", "").lower()
    metric_tag = f"{model_tag}_{metric}"
    spectrum_tag = f"{model_tag}_spectrum"

    metric_path = f"{METRICS_PATH}/{array_type}/{metric}/{model_name}/{num_layers}l"
    spectrum_path = f"{SPECTRUMS_PATH}/{array_type}/{model_name}/{num_layers}l"

    # Add an amplitude and bin threshold identifier to the metric tag if the metric is 'rmse' or 'detection_rate'
    if metric in ['rmse', 'detection_rate']:
        metric_tag = f"{metric_tag}_{amp_threshold}at_{bin_threshold}bt"
        metric_path = f"{metric_path}/{amp_threshold}at_{bin_threshold}bt"

   
    os.makedirs(metric_path, exist_ok=True)
    torch.save(metric_dict, f"{metric_path}/{metric_tag}.pt")

    os.makedirs(spectrum_path, exist_ok=True)
    torch.save(spectrums_dict, f"{spectrum_path}/{spectrum_tag}.pt")

    print(
        f"Metric results saved under '{metric_path}' with the metric tag: {metric_tag} \n"
        f"Spectrum results saved under '{spectrum_path}' with the spectrum tag: {spectrum_tag}"
    )

