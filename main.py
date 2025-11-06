import argparse
import json


from datasets.dataset_generator import generate_dictionary, generate_dataset_train, generate_dataset_test
from utils.training_utils import train_model
from utils.utils import set_random_seeds
from utils.metric_utils import evaluate_model

SEED=1234


def main():
    
    
    parser = argparse.ArgumentParser(description="Main script to manage dataset generation, training, and testing of models.")
    parser.add_argument('--config', type=str, help='Path to JSON config file.')
    subparsers = parser.add_subparsers(dest='command', help="Choose between dataset generation, training, and testing.")

    parser_array = subparsers.add_parser("create-array", help="Generate an array and its dictionary.")
    parser_array.add_argument('--array_type', type=str, default='SLA', help='Type of array (ULA or SLA).')
    parser_array.add_argument('--num_elements', type=int, default=30, help='Number of elements in the array.')
    parser_array.add_argument('--aperture', type=int, default=60, help='aperture of the array in lambda/2 units.')
    parser_array.add_argument('--dictionary_length', type=int, default=256, help='Length of the dictionary.')
   

    parser_dataset_train = subparsers.add_parser("create-trainset", help="Generate a dataset for training or testing.")
    parser_dataset_train.add_argument('--dictionary_path', type=str, help='Path to the dictionary.')
    parser_dataset_train.add_argument('--num_measurement_vectors', type=int, default=120000, help='Number of measurement vectors in the training set.')
    parser_dataset_train.add_argument('--max_number_sources', type=int, default=8, help='Maximum number of sources per measurement vector.')
    parser_dataset_train.add_argument('--snr', type=list, default=[15], help='Signal-to-noise ratio in dB.')
    parser_dataset_train.add_argument('--min_freq_separation_factor', type=int, default=1, help='Minimum frequency separation factor.')


    parser_dataset_test = subparsers.add_parser("create-testset", help="Generate a test dataset.")
    parser_dataset_test.add_argument('--dictionary_path', type=str, help='Path to the dictionary.')
    parser_dataset_test.add_argument('--snr_values', type=list, default=[0, 5, 10, 15, 20, 30, 35], help='List of SNR points (in dB) at which the model is tested.')
    parser_dataset_test.add_argument('--num_vectors_per_snr', type=int, default=1000, help='Number of test measurement vectors per SNR point.')
    parser_dataset_test.add_argument('--max_number_sources', type=int, default=8, help='Maximum number of sources per test measurement vector.')
    parser_dataset_test.add_argument('--min_freq_separation_factor', type=int, default=3, help='Minimum frequency separation factor.')
 
        
    parser_train = subparsers.add_parser("train-model", help="Train a model.")

    parser_train.add_argument('--model', type=str, default='CADMM-Net', help='Name of model to be trained:'
                        'ADMM-Net, THADMM-Net, CADMM-Net, CHADMM-Net, LISTA, TLISTA, THLISTA.')
                        
    parser_train.add_argument('--num_layers', type=int, default=30, help='Number of layers in the model.')
    parser_train.add_argument('--dataset_train_path', type=str, help='Path to the training set.')
    parser_train.add_argument('--epochs', type=int, default=30, help='Number of epochs.')
    parser_train.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser_train.add_argument('--batch_size', type=int, default=2048, help='Batch size.')
    parser_train.add_argument('--num_training_samples', type=int, default=1e5, help='''Number of measurement vectors to be used from the training set for the training.
                               The remaining vectors are used for validation.''')
    parser_train.add_argument('--model_path', type=str, help='Path to a specific model state to resume training from. This argument is optional.')
    parser_train.add_argument('--load_latest_state', type=str, default=True, help="Resumes training from latest saved state if available. Set to False to use a specific model state.")
    parser_train.add_argument('--device', type=str, default='cuda', help='Device to be used for training (cpu or cuda).')



    parser_metrics = subparsers.add_parser("evaluate-model", help="Evaluate the performance of a model or iterative method.")

    parser_metrics.add_argument('--model', type=str, default='ADMM-Net', help="Name of model or iterative method:" 
                        "ADMM-Net, THADMM-Net, CADMM-Net, 'CHADMM-Net', LISTA, TLISTA, THLISTA, ADMM, ISTA.")
    
    parser_metrics.add_argument('--num_layers', type=int, default=30, help='''Number of layers of model
                            or number of iterations in case of an iterative method.''')
    
    parser_metrics.add_argument('--dataset_test_path', type=str, help='Path to the test dataset to be used for the metric evaluation.')
    parser_metrics.add_argument('--model_path', type=str, help='Path to a specific model state to use for the metric evaluation. This argument is optional.')
    parser_metrics.add_argument('--load_latest_state', type=str, default=True, help='Loads the latest saved state if available. Set to False to use a specific model state.')

    parser_metrics.add_argument('--metric', type=str, default='detection_rate', help="Metric to be used for the model performance evaluation: detection_rate, nmse or rmse.")
    
    parser_metrics.add_argument('--bin_threshold', type=int, default=2, help='''Distance threshold for the detection rate metric. A candidate targets is considered for the next phase of detection if its bin index relative to the true target's is less than or equal to this value.''')
    
    parser_metrics.add_argument('--amp_threshold', type=float, default=0.4, help='''Amplitude threshold for the detection rate metric. A candidate targets is considered detected if it passed phase 1 and its amplitude relative to the true target's is greater than or equal to this value.''')
    parser_metrics.add_argument('--return_degs', type=bool, default=True, help="Returns the RMSE in degrees.")
    parser_metrics.add_argument('--device', type=str, default='cpu', help="Device to be used for performance evaluation (cpu or cuda).") 

    args = parser.parse_args()

    set_random_seeds(SEED)

    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            command_args = config.get(args.command, {})
            for key, value in command_args.items():
                setattr(args, key, value)     


    if args.command == 'create-array':
        generate_dictionary(args.array_type, args.num_elements, args.aperture, args.dictionary_length)
    elif args.command == 'create-trainset': 
        generate_dataset_train(args.dictionary_path, args.num_measurement_vectors, args.max_number_sources, args.snr, args.min_freq_separation_factor)
    elif args.command == 'create-testset':
        generate_dataset_test(args.dictionary_path, args.snr_values, args.num_vectors_per_snr, args.max_number_sources, args.min_freq_separation_factor)
    elif args.command == 'train-model':
        train_model(args.model, args.dataset_train_path, args.num_layers, args.epochs, args.lr, args.batch_size, args.num_training_samples, args.model_path, args.load_latest_state, args.device)
    elif args.command == 'evaluate-model':
        evaluate_model(args.model, args.dataset_test_path, args.num_layers, args.model_path, args.load_latest_state, args.metric, args.bin_threshold, args.amp_threshold, args.return_degs, args.device)

if __name__ == '__main__':
    main() 

