import os
import glob
import torch

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS_PATH = f"{BASE_PATH}/outputs/metrics"
LOSSES_PATH = f"{BASE_PATH}/outputs/losses"
SPECTRUMS_PATH = f"{BASE_PATH}/outputs/spectrums"


def load_output(output_type, load_all=False):
    """
    Reads output data for the given type.

    output_type: one of 'metrics', 'losses', or 'spectrums'
    load_all: if True, loads outputs from all model tags; if False (default),
              loads only the latest model tag based on modification time.

    Returns nested dictionary with the following structure:

    for output_type == 'spectrums' or 'losses'
        spectrums/losses[array_type][model_name][num_layers] 
    for output_type == 'metrics'
        metrics[array_type][metric][model_name][num_layers][thresh_folder] if 
            metric == 'rmse' or 'detection_rate'
        metrics[array_type][metric][model_name][num_layers] if metric == 'nmse'

    for load_all is True, the dictionary structure is identical to that above,
    but with an additional level for model_tag.
    """
    paths = {
        "metrics": METRICS_PATH,
        "losses": LOSSES_PATH,
        "spectrums": SPECTRUMS_PATH
    }
    base_dir = paths.get(output_type)
    if base_dir is None:
        raise ValueError(f"Invalid output type: {output_type}")

    results = {}
    # Loop over array_type directories
    for array_type in os.listdir(base_dir):
        array_path = os.path.join(base_dir, array_type)
        if not os.path.isdir(array_path):
            continue

        if output_type == "metrics":
            if array_type not in results:
                results[array_type] = {}
            # Loop over metric directories
            for metric in os.listdir(array_path):
                metric_path = os.path.join(array_path, metric)
                if not os.path.isdir(metric_path):
                    continue

                if metric not in results[array_type]:
                    results[array_type][metric] = {}

                # Loop over model directories
                for model_name in os.listdir(metric_path):
                    model_path = os.path.join(metric_path, model_name)
                    if not os.path.isdir(model_path):
                        continue

                    if model_name not in results[array_type][metric]:
                        results[array_type][metric][model_name] = {}

                    # Loop over num_layers folders
                    for layer_folder in os.listdir(model_path):
                        layer_path = os.path.join(model_path, layer_folder)
                        if not os.path.isdir(layer_path):
                            continue

                        num_layers = layer_folder.rstrip("l")
                        # For metrics "rmse" and "detection_rate", there is an extra thresh_folder level.
                        if metric in ["rmse", "detection_rate"]:
                            if num_layers not in results[array_type][metric][model_name]:
                                results[array_type][metric][model_name][num_layers] = {}
                            # Loop over threshold directories
                            for thresh_folder in os.listdir(layer_path):
                                thresh_path = os.path.join(layer_path, thresh_folder)
                                if not os.path.isdir(thresh_path):
                                    continue

                                if thresh_folder not in results[array_type][metric][model_name][
                                    num_layers
                                ]:
                                    results[array_type][metric][model_name][num_layers][
                                        thresh_folder
                                    ] = {}

                                pt_files = glob.glob(os.path.join(thresh_path, "*.pt"))
                                if not pt_files:
                                    continue

                                if not load_all:
                                    pt_file = max(pt_files, key=os.path.getmtime)
                                    try:
                                        data = torch.load(pt_file, weights_only=True)
                                        metric_value = data.get("average")
                                    except Exception as e:
                                        print(f"Error loading {pt_file}: {e}")
                                        continue
                                    # Directly assign the value without the extra model_tag key.
                                    results[array_type][metric][model_name][
                                        num_layers
                                    ][thresh_folder] = metric_value
                                else:
                                    for pt_file in pt_files:
                                        model_tag = os.path.splitext(
                                            os.path.basename(pt_file)
                                        )[0]
                                        try:
                                            data = torch.load(pt_file, weights_only=True)
                                            metric_value = data.get("average")
                                        except Exception as e:
                                            print(f"Error loading {pt_file}: {e}")
                                            continue
                                        results[array_type][metric][model_name][
                                            num_layers
                                        ][thresh_folder][model_tag] = metric_value
                        else:
                            # For nmse where there is no thresh_folder level.
                            if num_layers not in results[array_type][metric][model_name]:
                                results[array_type][metric][model_name][num_layers] = {}
                            pt_files = glob.glob(os.path.join(layer_path, "*.pt"))
                            if not pt_files:
                                continue

                            if not load_all:
                                pt_file = max(pt_files, key=os.path.getmtime)
                                try:
                                    data = torch.load(pt_file, weights_only=True)
                                    metric_value = data.get("average")
                                except Exception as e:
                                    print(f"Error loading {pt_file}: {e}")
                                    continue
                                # Directly assign the value without the extra model_tag key.
                                results[array_type][metric][model_name][num_layers] = metric_value
                            else:
                                for pt_file in pt_files:
                                    model_tag = os.path.splitext(
                                        os.path.basename(pt_file)
                                    )[0]
                                    try:
                                        data = torch.load(pt_file, weights_only=True)
                                        metric_value = data.get("average")
                                    except Exception as e:
                                        print(f"Error loading {pt_file}: {e}")
                                        continue
                                    results[array_type][metric][model_name][
                                        num_layers
                                    ][model_tag] = metric_value


                        results["snr"] = data.get("snr_values")
        else:
            # For losses and spectrums
            if array_type not in results:
                results[array_type] = {}

            for model_name in os.listdir(array_path):
                model_path = os.path.join(array_path, model_name)
                if not os.path.isdir(model_path):
                    continue

                if model_name not in results[array_type]:
                    results[array_type][model_name] = {}

                for layer_folder in os.listdir(model_path):
                    layer_path = os.path.join(model_path, layer_folder)
                    if not os.path.isdir(layer_path):
                        continue

                    num_layers = layer_folder.rstrip("l")
                    if num_layers not in results[array_type][model_name]:
                        results[array_type][model_name][num_layers] = {}

                    pt_files = glob.glob(os.path.join(layer_path, "*.pt"))
                    if not pt_files:
                        continue

                    if not load_all:
                        pt_file = max(pt_files, key=os.path.getmtime)
                        try:
                            data = torch.load(pt_file, weights_only=True)
                        except Exception as e:
                            print(f"Error loading {pt_file}: {e}")
                            continue

                        if output_type == "losses":
                            results[array_type][model_name][num_layers] = {
                                "training_loss": data.get("training_loss"),
                                "validation_loss": data.get("validation_loss"),
                            }
                        elif output_type == "spectrums":
                            results[array_type][model_name][num_layers] = {
                                "spectrums": data.get("spectrums"),
                                "ground_truth": data.get("ground_truth"),
                            }
                    else:
                        for pt_file in pt_files:
                            model_tag = os.path.splitext(os.path.basename(pt_file))[0]
                            try:
                                data = torch.load(pt_file, weights_only=True)
                            except Exception as e:
                                print(f"Error loading {pt_file}: {e}")
                                continue

                            if output_type == "losses":
                                results[array_type][model_name][num_layers][
                                    model_tag
                                ] = {
                                    "training_loss": data.get("training_loss"),
                                    "validation_loss": data.get("validation_loss"),
                                }
                            elif output_type == "spectrums":
                                results[array_type][model_name][num_layers][
                                    model_tag
                                ] = {
                                    "spectrums": data.get("spectrums"),
                                    "ground_truth": data.get("ground_truth"),
                                }
    return results