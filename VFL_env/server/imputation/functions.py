import torch.nn as nn
import torch
import os 
import json 
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

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

def save_metrics_ml(metrics_path,round_num,mse):
    os.makedirs(metrics_path, exist_ok=True)
    with open(os.path.join(metrics_path, "metrics.json"), "a") as f:
        json.dump({"round": round_num, "Global_MSE": mse}, f)
        f.write("\n")

def save_metrics(test_output,test_target,metrics_path, round_num):
    if test_output is None:
        return
    y_true = test_target
    y_pred = test_output.squeeze()
    mse = mean_squared_error(y_true, y_pred)
    print(f"Round {round_num} MSE: {mse}")

    # Ensure mse is a standard Python float
    mse = mse.item() if isinstance(mse, torch.Tensor) else float(mse)

    os.makedirs(metrics_path, exist_ok=True)
    with open(os.path.join(metrics_path, "metrics.json"), "a") as f:
        json.dump({"round": round_num, "MSE": mse}, f)
        f.write("\n")

def save_model(model_path,model):
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, "final_model.pth"))

# def save_model(self, model_dict):
#     model_path = "../results/final_model_weights.json"
#     with open(model_path, "w") as f:
#         json.dump(model_dict, f, indent=2)
#     print(f"Final model saved to {model_path}")

def preprocess_server_target_ml_r(data_path, target_col):
    data = pd.read_csv(data_path)
    if data[target_col].dtype in ['int64', 'float64'] :
        preprocessor = ColumnTransformer([('num', StandardScaler(),[target_col])])
        X = preprocessor.fit_transform(data)
        return torch.tensor(X.toarray() if hasattr(X, "toarray") else X, dtype=torch.float32).view(-1, 1)
 
    else : 
        raise ValueError(f"Target '{target_col}' is categorical.")
    
    
def preprocess_server_target(csv_path, target_feature, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)

    if df[target_feature].dtype in ['int64', 'float64']:
        target = df[target_feature].values
        y = torch.tensor(target, dtype=torch.float32).view(-1, 1)

        n = len(y)
        indices = np.arange(n)
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)

        return (
            y[train_indices],
            y[test_indices],
            train_indices,
            test_indices
            )
    else:
        target_f = pd.get_dummies(df[target_feature])
        y = pd.get_dummies(df[target_feature]).values
        label_map = list(target_f.columns)
        
        label_map_path = "../results/dl_classification/"  
        os.makedirs(label_map_path, exist_ok=True)

        # Save the label map
        with open( "../results/dl_classification/label_map.pkl", "wb") as f:
            pickle.dump(label_map, f)

    y_train, y_test = train_test_split(y, test_size=test_size, random_state=random_state)

    return (
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )


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
    
    # we apply it to orchestrate the sent and received index between nodes and the server
def reshape_list_with_none(numbers):
    max_index = max(numbers)
    new_list = [None] * (max_index + 1)
    for i, num in enumerate(numbers):
        new_list[i + (max_index + 1 - len(numbers))] = num
    return new_list


