# Circulant ADMM-Net for Fast High-resolution DoA Estimation

This repository supplements the [paper](https://arxiv.org/pdf/2502.19076) "Circulant ADMM-Net for Fast High-resolution DoA Estimation". CADMM-Net is a structured deep-unrolled network based on the alternating direction method of multipliers (ADMM) algorithm. The only learnable parameters of the network are the (real-valued) eigenvalues in the eigen-decomposition of a circulant matrix in each layer, resulting in a considerable decrease in training and inference times through FFTs while achieving a competitive performance against ADMM-Net. 

Below is a guided walkthrough to replicate
the experimental results.
## Table of Contents

1. [Repository Setup](#repository-setup)
2. [Datasets Setup](#datasets-setup)
   - [Dictionary Generation](#dictionary-generation)
   - [Training Dataset](#training-dataset)
   - [Test Dataset](#test-dataset)
3. [Model Training](#model-training)
4. [Metric Evaluation](#metric-evaluation)
5. [Additional Resources](#additional-resources)

## Repository Setup

We first setup the repo and environment.

```sh
git clone https://github.com/youvalklioui/cadmmnet.git
cd cadmmnet

conda env create -f environment.yml
conda activate cadmmnet_env
```
## Datasets Setup
### Dictionary Generation
We generate a dictionary for the experimental setup by specifying an `array_type` (`'SLA'` or `'ULA'`), the `aperture` (in $\lambda/2$ units), the  `num_elements` to be randomly sub-sampled along with the `dictionary_length` for the sparse recovery problem.

```sh
python main.py create-array \
  --array_type 'SLA' \
  --num_elements 30 \
  --aperture 60 \
  --dictionary_length 256
```

The example above will create and save a `dictionary_30elem_60ap_256len.pt` file under `./datasets/sla`. Using this dictionary, we can then create a training dataset and, optionally, a test dataset for the metrics evaluation part.

### Training Dataset
 For generating the training dataset, we run

```sh
python main.py create-trainset \
  --dictionary_path './datasets/sla/dictionary_30elem_60ap_256len.pt' \
  --num_measurement_vectors 120000 \
  --max_number_sources 8 \
  --snr [15]\
  --min_freq_separation_factor 1
```
This will generate a `dataset_train_8tgts_15dbsnr_1fres.pt` file under `./datasets/sla`. `snr` can be either a single or two element list. For the single-element case, as above, the SNR is fixed for all the measurement vectors. If it is a two-element list, with `[20, 40]` for example, then the SNR will be randomly generated from this interval for each measurement vector. `min_freq_separation_factor` uses `num_elements`to specify how close the generated frequencies can be, which is given by `1/(min_freq_separation_factor * num_elements)`.  

### Test Dataset
For generating the test dataset, we run

```sh
python main.py create-testset \
  --dictionary_path './datasets/sla/dictionary_30elem_60ap_256len.pt' \
  --snr_values [0, 5, 10, 15, 20, 25, 30, 35] \
  --num_vectors_per_snr 1000 \
  --max_number_sources 8 \
  --min_freq_separation_factor 3 \
```
which will create a `dataset_test_8tgts_0to35dbsnr_3fres.pt` under the same directory as the training set. The datasets can also be download from [here](https://zenodo.org/records/14926792) for the SLA setting or [here](https://zenodo.org/records/14926980) for the ULA setting, and need to be placed under `./datasets/sla` or `./datasets/ula`, respectively. In the example above, we will have `1000` measurement test vectors for each value in `snr_values`.

## Model Training
 We can then train our model with

```sh
python main.py train \
    --model 'CADMM-Net' \
    --num_layers 15 \
    --dataset_train_path './datasets/sla/dataset_train_8tgts_15dbsnr_1fres.pt' \
    --epochs 30 \
    --lr 0.0001 \
    --batch_size 2048 \
    --num_training_samples 100000 \
    --model_path null \
    --load_latest_state false \
    --device 'cuda'
```
We use `--num_training_samples` to indicate the number of measurement vectors to be used from the training set for the training proper, the rest will be used for computing the validation loss. If there is an already saved model state, we can pass its path using `--model_path` to resume the training. The models currently supported are `'ADMM-Net'`, `'CADMM-Net'`, `'CHADMM-Net'`, `'LISTA'`, `'TLISTA'`, and `'THLISTA'`. Once through, the model state will then be saved under `./models/states/sla` and the training/validation loss is saved under `./outputs/losses/sla`. The models weights can be directly downloaded from [here](https://zenodo.org/records/14927063) for the SLA setting or [here](https://zenodo.org/records/14927193) for the ULA , and need to be placed in the `./models/states/sla` or '`./models/states/sla`' subdirectory, respectively.

## Metric Evaluation
We can then evaluate the model's performance with 

```sh
python main.py evaluate-model \
  --model 'CADMM-Net' \
  --num_layers 15 \
  --dataset_test_path './datasets/sla/dataset_test_8tgts_0to35dbsnr_3fres.pt' \
  --model_path null \
  --load_latest_state true \
  --metric 'detection_rate' \
  --bin_threshold 2 \
  --amp_threshold 0.4 \
  --return_degs true \
  --device cpu
```

This will load the latest model state for the given model name and number of layers, and will evaluate its performance with respect to the specified metric, which can be ` 'detection_rate' `, `'rmse'`, or `'nmse'`. The `amp_threshold` and `bin_threshold` options are ignored when `metric` is set to `'nmse'`. Additional details on the metrics computation and these thresholds can be found in the docstrings of [metric_utils.py](utils/metric_utils.py). We can also specify a fixed point method for `model`, such as `'ADMM'` or `'ISTA'`, in which case `num_layers` corresponds to the number of iterations. The resulting metric will then be saved as a `.pt` file under `./outputs/metrics/sla` and the estimated spectrums under `./output/spectrums/sla`.

## Acknowledgement
This work was funded by the RAISE collaboration framework between
Eindhoven University of Technology and NXP Semiconductors, including a PPS-supplement
from the Dutch Ministry of Economic Affairs and Climate Policy.


