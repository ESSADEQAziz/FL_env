import os
import json
import torch
import torch.nn as nn
from flwr.common import parameters_to_ndarrays


def save_model(aggregated_parameters,server_round,input_dim,indx):
        
        if indx == "dl" :
            save_path=f"../results/dl_results/agg_model/model_round{server_round}.pth"
            # Load into model and save
            model = SimpleRegressor(input_dim=input_dim) 
        elif indx == "ml" :
            save_path=f"../results/ml_results/agg_model/model_round{server_round}.pth"
            # Load into model and save
            model = LinearRegressionModel(input_dim=input_dim) 
        else :
            raise ValueError(f"Failure during saving the model.")
        
        # Save to specified path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert parameters to PyTorch model state_dict
        model_state_dict = parameters_to_ndarrays(aggregated_parameters)

        model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), model_state_dict)})
            
        torch.save(model.state_dict(), save_path)



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
    
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  

def save_metrics(history,indx):
    if indx == "dl":
        path = "../results/dl_results/metrics.json"
    elif indx == "ml" :
        path = "../results/ml_results/metrics.json"
    else :
        raise ValueError("Failure during saving metrics.")
        
    
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



