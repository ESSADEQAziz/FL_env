import os
import json
import torch
import torch.nn as nn
import numpy as np
from flwr.common import parameters_to_ndarrays

def save_model(aggregated_parameters, server_round, input_dim, indx, num_classes=0):
    """
    Save the model state_dict with weights from aggregated parameters.
    
    Args:
        aggregated_parameters: Aggregated model parameters from Flower
        server_round: Current round number
        input_dim: Input dimension for the model
        indx: Model type identifier ('dl_r', 'ml_r', 'dl_c', 'ml_c')
        num_classes: Number of classes for classification models
    """
    # Determine save path and model type based on index
    if indx == "dl_r":
        save_path = "../results/dl_results/regression/agg_model/model_round.pth"
        model = SimpleRegressor(input_dim=input_dim) 
    elif indx == "ml_r":
        save_path = "../results/ml_results/regression/agg_model/model_round.pth"
        model = LinearRegressionModel(input_dim=input_dim) 
    elif indx == "ml_c":
        save_path = "../results/ml_results/classification/agg_model/model_round.pth"
        model = LogisticRegressionModel(input_dim=input_dim, output_dim=num_classes) 
    elif indx == "dl_c":
        save_path = "../results/dl_results/classification/agg_model/model_round.pth"
        model = SimpleClassifier(input_dim=input_dim, num_classes=num_classes)
    else:
        raise ValueError("Failure during saving the model: invalid index '{indx}'")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert parameters to NumPy arrays
    model_params = parameters_to_ndarrays(aggregated_parameters)
    
    # Load parameters into model
    model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), model_params)})
    
    # Save the state_dict (safer and more compatible)
    torch.save(model.state_dict(), save_path)
    
    print(f"Model state_dict saved to {save_path}")
    
    # Save model architecture information separately for easier loading later
    model_info = {
        "model_type": indx,
        "input_dim": input_dim,
        "num_classes": num_classes if indx in ["dl_c", "ml_c"] else 0,
        "round": server_round
    }
    
    info_path = os.path.join(
        os.path.dirname(save_path),
        "model_info_round.json"
    )
    
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    return save_path   

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)    

class SimpleRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1):
        super(SimpleRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  

def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    else:
        return obj

def save_history_metrics(history, indx):

    isClassification = True
    # Determine path and classification flag
    if indx == "dl_r":
        path = "../results/dl_results/regression/metrics.json"
        isClassification = False

    elif indx == "ml_r":
        path = "../results/ml_results/regression/metrics.json"
        isClassification = False

    elif indx == "ml_c":

        path = "../results/ml_results/classification/metrics.json"

    elif indx == "stat":
        path = "../results/stat_results/metrics.json"

        os.makedirs(os.path.dirname(path), exist_ok=True)

        entry = {"round": 1}

        # Get the latest round's 'aggregated_mean_med' values
        if "aggregated_mean_med" in history.metrics_distributed_fit:
            # Assume it's a list of (round, array) pairs
            last_entry = history.metrics_distributed_fit["aggregated_mean_med"][-1]
            _, values = last_entry  # unpack (round, ndarray)
            entry["Aggregated mean/median"] = make_json_serializable(values)

        with open(path, 'w') as f:
            json.dump(entry, f, indent=2)
        
        return True

    elif indx == "dl_c":
        path = "../results/dl_results/classification/metrics.json"

    else:
        raise ValueError("Failure during saving metrics: invalid `indx` value.")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Collect all rounds
    rounds = set()
    
    # Add rounds from distributed losses
    for round_num, _ in history.losses_distributed:
        rounds.add(round_num)
    
    # Add rounds from centralized losses
    for round_num, _ in history.losses_centralized:
        rounds.add(round_num)
    
    # Add rounds from metrics
    for metric, values in history.metrics_distributed_fit.items():
        for round_num, _ in values:
            rounds.add(round_num)
    
    for metric, values in history.metrics_distributed.items():
        for round_num, _ in values:
            rounds.add(round_num)
    
    for metric, values in history.metrics_centralized.items():
        for round_num, _ in values:
            rounds.add(round_num)
    
    # Sort rounds
    sorted_rounds = sorted(rounds)
    
    # Create metrics entries for each round
    metrics_entries = []
    
    for round_num in sorted_rounds:
        entry = {"round": round_num}
        
        # Find distributed loss for this round
        distributed_loss = 0.0
        for r, loss in history.losses_distributed:
            if r == round_num:
                distributed_loss = loss
                break
        entry["distributed_loss"] = distributed_loss
        
        # Find centralized loss for this round
        centralized_loss = 0.0
        for r, loss in history.losses_centralized:
            if r == round_num:
                centralized_loss = loss
                break
        entry["centralized_loss"] = centralized_loss            
        
        if isClassification : 
            # Find distributed accuracy for this round
            distributed_accuracy = 0.0
            if "accuracy" in history.metrics_distributed_fit:
                for r, acc in history.metrics_distributed_fit["accuracy"]:
                    if r == round_num:
                        distributed_accuracy = acc
                        break
            entry["distributed_accuracy"] = distributed_accuracy
            
            # Find centralized accuracy for this round
            centralized_accuracy = 0.0
            if "accuracy" in history.metrics_centralized:
                for r, acc in history.metrics_centralized["accuracy"]:
                    if r == round_num:
                        centralized_accuracy = acc
                        break
            entry["centralized_accuracy"] = centralized_accuracy
        
        metrics_entries.append(entry)
    
    # Write to file, one JSON object per line
    with open(path, 'w') as f:
        for entry in metrics_entries:
            f.write(json.dumps(entry) + '\n')
    return True




