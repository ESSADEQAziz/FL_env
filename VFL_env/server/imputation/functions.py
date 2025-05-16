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

           
def save_model(model, client_encoders=None, model_type="dl_r"):
    """
    Save the complete VFL model (server model + client encoders) for future predictions
    
    Args:
        model: The server model
        client_encoders: Dictionary of client encoders {node_id: encoder}
        model_type: Type of model ("dl_r", "dl_c", "ml_r", "ml_c")
    """
    # Map model type to directory
    model_type_mapping = {
        "dl_r": "dl_regression",
        "dl_c": "dl_classification",
        "ml_r": "ml_regression",
        "ml_c": "ml_classification"
    }
    
    model_path = f"../results/{model_type_mapping.get(model_type, 'dl_regression')}"
    os.makedirs(model_path, exist_ok=True)
    
    # Save complete server model (not just state_dict)
    torch.save(model, os.path.join(model_path, "final_model.pth"))
    
    # For backward compatibility, also save state_dict
    torch.save(model.state_dict(), os.path.join(model_path, "final_model_state_dict.pth"))
    
    # Save model metadata
    model_info = {
        "model_type": model_type,
        "input_dim": model.model[0].in_features if hasattr(model, 'model') else None,
        "output_dim": model.model[-1].out_features if hasattr(model, 'model') else None,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open(os.path.join(model_path, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Save client encoders if provided
    if client_encoders:
        # Ensure encoder directory exists
        encoder_dir = os.path.join(model_path, "encoders")
        os.makedirs(encoder_dir, exist_ok=True)
        
        # Save each encoder
        for node_id, encoder in client_encoders.items():
            encoder_path = os.path.join(model_path, f"client_encoder_{node_id}.pth")
            torch.save(encoder, encoder_path)
        
        print(f"Saved server model and {len(client_encoders)} client encoders to {model_path}")
    else:
        print(f"Saved server model to {model_path}")
    
    return model_path

def preprocess_server_target_ml_r(data_path, target_col):
    data = pd.read_csv(data_path)
    if data[target_col].dtype in ['int64', 'float64'] :
        target_mean = data[target_col].mean()
        data[target_col].fillna(target_mean, inplace=True)
        # Replaced NaN values in target with mean: target_mean
        preprocessor = ColumnTransformer([('num', StandardScaler(),[target_col])])

        preprocessor_path = "../results/ml_regression/server_preprocessor/"
        os.makedirs(preprocessor_path, exist_ok=True)

        with open(f"{preprocessor_path}target_scaler.pkl", "wb") as f:
            pickle.dump(preprocessor, f)

        Y = preprocessor.fit_transform(data)
        return torch.tensor(Y.toarray() if hasattr(Y, "toarray") else Y, dtype=torch.float32).view(-1, 1)
 
    else : 
        raise ValueError(f"Target '{target_col}' is categorical.")
    
def preprocess_server_target_ml_c(data_path, target_col,test_size=0.2,random_state=42):
    data = pd.read_csv(data_path)
    if data[target_col].dtype in ['object'] :

        data[target_col].fillna(data[target_col].mode()[0], inplace=True)
        # Replaced NaN values in categorical target with the most frequent value 'mode'


        target_f = pd.get_dummies(data[target_col])
        y = pd.get_dummies(data[target_col]).values
        label_map = list(target_f.columns)

        label_map_path = "../results/ml_classification/server_preprocessor"  
        os.makedirs(label_map_path, exist_ok=True)

        # Save the label map
        with open( f"{label_map_path}label_map.pkl", "wb") as f:
            pickle.dump(label_map, f)

        y_train, y_test = train_test_split(y, test_size=test_size, random_state=random_state)

        return (
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        )
 
    else : 
        raise ValueError(f"Target '{target_col}' is numerical.")
    
    
def preprocess_server_target(csv_path, target_feature,approche, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)

        # Save the features processor for the future tests and evaluations 
    if approche == 'dl_r':
        preprocessor_path = "../results/dl_regression/server_preprocessor/"
    elif approche == 'dl_c':
        preprocessor_path = "../results/dl_classification/server_preprocessor/"
    elif approche == 'ml_r':
        preprocessor_path = "../results/ml_regression/server_preprocessor/"
    elif approche == 'ml_c':
        preprocessor_path = "../results/ml_classification/server_preprocessor/"

    os.makedirs(preprocessor_path, exist_ok=True)

    if df[target_feature].dtype in ['int64', 'float64']:

        target_mean = df[target_feature].mean()
        df[target_feature].fillna(target_mean, inplace=True)
        # Replaced NaN values in target with mean: target_mean

        preprocessor = ColumnTransformer([('num', StandardScaler(),[target_feature])])
        with open(f"{preprocessor_path}target_scaler.pkl", "wb") as f:
            pickle.dump(preprocessor, f)
        Y = preprocessor.fit_transform(df)
        y = torch.tensor(Y.toarray() if hasattr(Y, "toarray") else Y, dtype=torch.float32).view(-1, 1)

        n = len(y)
        indices = np.arange(n)
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)

        return (
            y[train_indices],
            y[test_indices],
            train_indices,
            test_indices
            )
    
    df[target_feature].fillna(df[target_feature].mode()[0], inplace=True)
    # Replaced NaN values in categorical target with the most frequent value 'mode'

    target_f = pd.get_dummies(df[target_feature])
    y = pd.get_dummies(df[target_feature]).values
    label_map = list(target_f.columns)
        
    os.makedirs(preprocessor_path, exist_ok=True)

    # Save the label map
    with open( "../results/dl_classification/server_preprocessor/label_map.pkl", "wb") as f:
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
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
    # we apply it to orchestrate the sent and received index between nodes and the server


def reshape_list_with_none(numbers):
    max_index = max(numbers)
    new_list = [None] * (max_index + 1)
    for i in numbers :
        new_list[i]=i
    return new_list

class ClientEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=4):
        super(ClientEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)
  

  