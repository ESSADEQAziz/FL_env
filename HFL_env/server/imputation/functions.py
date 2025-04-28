import os
import numpy as np
import json
import torch
import torch.nn as nn
from flwr.common import parameters_to_ndarrays



def evaluate_ml_values(history, strategy_param, logger ):

    results_path = "../results/ml_results/metrics.json"
    # Convert to dict for easy lookup
    losses = dict(history.losses_distributed)
    param_log = dict(strategy_param)

    all_metrics = []  # List to accumulate all metrics for each round

    # Loop through rounds
    for round_num, mse in losses.items():
        parameters = param_log.get(round_num, [0.0, 0.0])  # fallback if missing

        # Handle scalar extraction safely
        def safe_to_float(x):
            if isinstance(x, np.ndarray):
                return x.item() if x.size == 1 else float(x[0])
            return float(x)

        a = safe_to_float(parameters[0])
        b = safe_to_float(parameters[1])
        mse = float(mse)

        metrics_entry = {
            "round": round_num,
            "parameters": {
                "a": a,
                "b": b
            },
            "global_mse": mse
        }

        # Append each round's metrics to the list
        all_metrics.append(metrics_entry)

    # Ensure the directory exists for saving the results
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Save all metrics into a single JSON file
    try:
        with open(results_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"[✓] Saved: {results_path}")
    except Exception as e:
        logger.error(f"[✗] Failed to write {results_path}: {e}")

def save_dl_model(aggregated_parameters,server_round,total_rounds,save_path):
        # Save model only after final round
        if server_round == total_rounds:  # Define this during init or use a constant
            # Convert parameters to PyTorch model state_dict
            model_state_dict = parameters_to_ndarrays(aggregated_parameters)

            # Load into model and save
            model = SimpleRegressor()  # Replace with your actual model class
            model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), model_state_dict)})

            # Save to specified path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            return True



class SimpleRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1):
        super(SimpleRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)
    
def evaluate_dl_values(history):

    path="../results/dl_results/metrics.json"
    # Prepare directory
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert history to dictionary
    history_dict = {
        "loss": history.losses_distributed,  # list of tuples (round, loss)
        "metrics": history.metrics_centralized  # dict of metric_name -> list of (round, value)
    }

    # Save to JSON
    with open(path, "w") as f:
        json.dump(history_dict, f, indent=4)

    return history

def evaluate_dl_values_2(history):
    path = "../results/dl_results/metrics.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    distributed = history.losses_distributed
    centralized = history.losses_centralized # Replace "loss" with the right key if needed

    with open(path, "w") as f:
        for i, (round_num, distributed_loss) in enumerate(distributed):
            record = {"round": round_num, "distributed_loss": distributed_loss}
            if i < len(centralized):
                record["centralized_loss"] = centralized[i][1]  # (round, value)
            json.dump(record, f)
            f.write("\n")

    return history

def insure_none(x):
    if torch.isnan(x).any():
        print("Warning: NaN values detected in target y, replacing them with 0 to avoid loss and gradients calculus.(consider it as noise)")
        x = torch.nan_to_num(x, nan=0.0)
    return x 



