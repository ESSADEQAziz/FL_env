import os
import json
import torch
import torch.nn as nn
from flwr.common import parameters_to_ndarrays


def save_model(aggregated_parameters,server_round,input_dim,indx,num_classes=0):
        
        if indx == "dl_r" :
            save_path=f"../results/dl_results/regression/agg_model/model_round{server_round}.pth"
            # Load into model and save
            model = SimpleRegressor(input_dim=input_dim) 
        elif indx == "ml_r" :
            save_path=f"../results/ml_results/regression/agg_model/model_round{server_round}.pth"
            # Load into model and save
            model = LinearRegressionModel(input_dim=input_dim) 
        elif indx == "ml_c" :
            save_path=f"../results/ml_results/classification/agg_model/model_round{server_round}.pth"
            # Load into model and save
            model = LogisticRegressionModel(input_dim=input_dim,output_dim=num_classes) 
        elif indx == "dl_c" :
            save_path=f"../results/dl_results/classification/agg_model/model_round{server_round}.pth"
            # Load into model and save
            model = SimpleClassifier(input_dim=input_dim, num_classes=num_classes)
        else :
            raise ValueError(f"Failure during saving the model.")
        
        # Save to specified path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert parameters to PyTorch model state_dict
        model_state_dict = parameters_to_ndarrays(aggregated_parameters)

        model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), model_state_dict)})
            
        torch.save(model.state_dict(), save_path)


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
            nn.Linear(hidden_dim, hidden_dim),
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

def save_metrics(history, indx):
    # Determine path and classification flag
    if indx == "dl_r":
        path = "../results/dl_results/regression/metrics.json"
        is_classification = False
    elif indx == "ml_r":
        path = "../results/ml_results/regression/metrics.json"
        is_classification = False
    elif indx == "ml_c":
        path = "../results/ml_results/classification/metrics.json"
        is_classification = True
    elif indx == "dl_c":
        path = "../results/dl_results/classification/metrics.json"
        is_classification = True
    else:
        raise ValueError("Failure during saving metrics: invalid `indx` value.")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Prepare loss and accuracy data as dictionaries for easy access
    dist_loss_dict = dict(history.losses_distributed)
    cent_loss_dict = dict(history.losses_centralized)

    dist_acc_dict = {}
    cent_acc_dict = {}

    if is_classification:
        dist_acc_dict = dict(history.metrics_distributed.get("accuracy", []))
        cent_acc_dict = dict(history.metrics_centralized.get("accuracy", []))

    # Get all rounds from distributed loss keys
    rounds = sorted(dist_loss_dict.keys())

    with open(path, "w") as f:
        for round_num in rounds:
            record = {
                "round": round_num,
                "distributed_loss": float(dist_loss_dict.get(round_num, 0.0)),
                "centralized_loss": float(cent_loss_dict.get(round_num, 0.0))
            }

            if is_classification:
                record["distributed_accuracy"] = float(dist_acc_dict.get(round_num, 0.0))
                record["centralized_accuracy"] = float(cent_acc_dict.get(round_num, 0.0))

            json.dump(record, f)
            f.write("\n")

    return True



def insure_none(x):
    if torch.isnan(x).any():
        print("Warning: NaN values detected in target y, replacing them with 0 to avoid loss and gradients calculus.(consider it as noise)")
        x = torch.nan_to_num(x, nan=0.0)
    return x 



