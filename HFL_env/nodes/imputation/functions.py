import pandas as pd
import numpy as np
import os
import math
import json
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def evaluate_statistical_values(reference_json_path, local_parameters, aggegated_parameters,NODE_ID):

    results_path= f'../results/statistical_results/metrics_node_{NODE_ID}.json'
    
        # Ensure the directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Load the reference data
    with open(reference_json_path, 'r') as f:
        reference_data = json.load(f)
    
    # Extract the original values
    original_values = list(reference_data["missing_values"].values())
    original_values=remove_nan_values(original_values)
    
    # Initialize counters [local_mean,agg_mean,local_med,agg_med]
    total_values=len(original_values)
    closer_to=[0]*4
    diffs=[0]*4
    
    # Compute absolute differences for each original value
    for original_val in original_values:

        diffs[0] = abs(original_val - local_parameters[0] )
        diffs[1] = abs(original_val - aggegated_parameters[0])

        diffs[2] = abs(original_val - local_parameters[0] )
        diffs[3] = abs(original_val - aggegated_parameters[1])
        
        # For the mean
        if diffs[0] < diffs[1]:
            closer_to[0] += 1
        elif diffs[1] < diffs[0]: 
            closer_to[1] += 1
        else:
            # If equidistant, count for both
            closer_to[0] += 0.5
            closer_to[1] += 0.5

        # For the Median    
        if diffs[2] < diffs[3]:
            closer_to[2] += 1
        elif diffs[3] < diffs[2]: 
            closer_to[3] += 1
        else:
            closer_to[2] += 0.5
            closer_to[3] += 0.5
        
    
    # Calculate accuracy (percentage of values closer to each value)
    accuracy=[0]*4
    accuracy[0] = (closer_to[0] / total_values) * 100 if total_values > 0 else 0
    accuracy[1] = (closer_to[1] / total_values) * 100 if total_values > 0 else 0    

    accuracy[2] = (closer_to[2] / total_values) * 100 if total_values > 0 else 0
    accuracy[3] = (closer_to[3] / total_values) * 100 if total_values > 0 else 0
    
    # Calculate precision (inverse of mean squared error)
    mse=[0]*4
    mse[0] = np.mean([(v - local_parameters[0])**2 for v in original_values]) if original_values else 0
    mse[1] = np.mean([(v - aggegated_parameters[0])**2 for v in original_values]) if original_values else 0

    mse[2] = np.mean([(v - local_parameters[1])**2 for v in original_values]) if original_values else 0
    mse[3] = np.mean([(v - local_parameters[1])**2 for v in original_values]) if original_values else 0    


    # Convert any numpy values to native Python types
    for i in range(4):
        if isinstance(accuracy[i], np.ndarray) or isinstance(accuracy[i], np.number):
            accuracy[i] = accuracy[i].item()
        if isinstance(mse[i], np.ndarray) or isinstance(mse[i], np.number):
            mse[i] = mse[i].item()
        if isinstance(closer_to[i], np.ndarray) or isinstance(closer_to[i], np.number):
            closer_to[i] = closer_to[i].item()
            
    # Ensure all local_parameters and aggegated_parameters values are JSON serializable
    for i in range(len(local_parameters)):
        if isinstance(local_parameters[i], np.ndarray) or isinstance(local_parameters[i], np.number):
            local_parameters[i] = local_parameters[i].item()
    
    for i in range(len(aggegated_parameters)):
        if isinstance(aggegated_parameters[i], np.ndarray) or isinstance(aggegated_parameters[i], np.number):
            aggegated_parameters[i] = aggegated_parameters[i].item()

    # Return the results
    data = {
        "local_mean": {
            "value": local_parameters[0],
            "accuracy": accuracy[0],
            "mse": mse[0],
            "closer_count": closer_to[0]
        },
        "aggregated_mean": {
            "value": aggegated_parameters[0],
            "accuracy": accuracy[1],
            "mse": mse[1],
            "closer_count": closer_to[1]
        },
        "local_median": {
            "value": local_parameters[1],
            "accuracy": accuracy[2],
            "mse": mse[2],
            "closer_count": closer_to[2]
        },
        "aggregated_median": {
            "value": aggegated_parameters[1],
            "accuracy": accuracy[3],
            "mse": mse[3],
            "closer_count": closer_to[3]
        },
        "total_values": total_values
    }
        # Save to JSON file
    with open(results_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

# This function aims to create some random missing data within a table based on a given missing_rate 

def evaluate_ml_values(node_id, aggregated_parameters, local_mse, round=0):

    results_path = f'../results/ml_results/metrics_node_{node_id}.json'
    
    # Convert NumPy arrays to Python native types
    def numpy_to_python(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [numpy_to_python(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: numpy_to_python(v) for k, v in obj.items()}
        else:
            return obj
    
    # Convert parameters to JSON-serializable format
    try:
        if hasattr(aggregated_parameters, 'tensors'):
            # Handle Flower parameters format
            from flwr.common.parameter import parameters_to_ndarrays
            params_list = parameters_to_ndarrays(aggregated_parameters)
            if len(params_list) >= 2:
                parameters = [params_list[0], params_list[1]]
            else:
                parameters = params_list
        else:
            # Handle direct numpy arrays or lists
            parameters = aggregated_parameters
            
        # Convert to Python native types
        parameters = numpy_to_python(parameters)
        
        # Make sure we have a and b parameters
        if isinstance(parameters, list) and len(parameters) >= 2:
            a = parameters[0]
            b = parameters[1]
        else:
            print(f"Unexpected parameter format: {parameters}. Using default values.")
            a = 0.0
            b = 0.0
            
    except Exception as e:
        print(f"Error processing parameters: {e}")
        a = 0.0
        b = 0.0
    
    # Convert MSE values to Python native types
    local_mse = numpy_to_python(local_mse)
    # Ensure the directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Prepare the metrics for this round
    metrics_entry = {
        "node_id": node_id,
        "round": round,
        "parameters": {
            "a": parameters[0],
            "b": parameters[1]
        },
        "local_mse": float(local_mse),
    }
    
    # Load existing data or create new file
    try:
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                data = json.load(f)
                # Check if data is a list or a single record
                if not isinstance(data, list):
                    data = [data]
        else:
            data = []
    except json.JSONDecodeError:
        print(f"Could not decode existing JSON at {results_path}. Creating new file.")
        data = []
    except Exception as e:
        print(f"Error reading existing metrics file: {e}")
        data = []
    
    # Add new metrics
    data.append(metrics_entry)
    
    # Write back to file
    try:
        with open(results_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully updated metrics for node {node_id} at round {round}")
    except Exception as e:
        print(f"Error writing metrics to file: {e}")


def create_missing_values(csv_path: str, feature_name: str, missing_rate: float, 
                          node_id) -> str:
    output_dir = f"../missing_data/node{node_id}"
    # Validate inputs
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if missing_rate < 0 or missing_rate > 1:
        raise ValueError("Missing rate must be between 0 and 1")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Check if feature exists in the dataset
    if feature_name not in df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in the dataset")
    
    # Calculate the number of values to set as missing
    num_samples = len(df)
    num_missing = int(num_samples * missing_rate)
    
    # Randomly select indices to set as missing
    missing_indices = np.random.choice(num_samples, num_missing, replace=False)
    
    # Store the original values and their indices
    original_values = {
        "feature_name": feature_name,
        "missing_rate": missing_rate,
        "missing_values": {
            str(idx): float(df.loc[idx, feature_name]) 
            for idx in missing_indices
        }
    }
    
    # Save the original values to a JSON file
    file_basename = os.path.basename(csv_path).replace(".csv", "")
    reference_path = os.path.join(
        output_dir, 
        f"{file_basename}_{feature_name}_missing{int(missing_rate*100)}.json"
    )
    
    with open(reference_path, 'w') as f:
        json.dump(original_values, f, indent=2)
    
    # Create a copy of the dataframe with missing values
    df_missing = df.copy()
    df_missing.loc[missing_indices, feature_name] = np.nan
    
    # Save the modified dataframe
    modified_csv_path = os.path.join(
        output_dir, 
        f"{file_basename}_{feature_name}_missing{int(missing_rate*100)}.csv"
    )
    df_missing.to_csv(modified_csv_path, index=False)
    
    return reference_path

def calculate_statistics(file_path, feature):
    df = pd.read_csv(file_path)
    mean_value = df[feature].mean()
    median_value = df[feature].median()
    return [mean_value, median_value]

def remove_nan_values(lst):
    return [x for x in lst if not (isinstance(x, float) and math.isnan(x))]

#-----------------------------------------------------------------------------------------------------------------

def preprocess_node_data(csv_path,features,target,indx):
    df = pd.read_csv(csv_path)
    column_names = df.columns

    num_features=[]
    cat_features=[]

    for item in column_names:
            if item in features :
                if df[item].dtype in ['int64', 'float64'] :
                    num_features.append(item)
                elif df[item].dtype in ['object']:
                    cat_features.append(item)

    preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
            ])

    if df[target].dtype in ['int64', 'float64']:
        # Preprocessor for target
        target_scaler = StandardScaler()
        Y = target_scaler.fit_transform(df[[target]])  # Keep as 2D array (n_samples, 1)

    elif df[target].dtype in ['object']:
        target_f = pd.get_dummies(df[target])
        Y = pd.get_dummies(df[target]).values
        label_map = list(target_f.columns)
        
    else :
        raise ValueError("Failure processing taget feature.")
        

    if indx == 'dl_r':
        # Save the features processor for the future tests and evaluations 
        preprocessor_path = "../results/dl_regression/"  
        os.makedirs(preprocessor_path, exist_ok=True)
        # Save target scaler separately
        with open(os.path.join(preprocessor_path, "target_scaler.pkl"), "wb") as f:
            pickle.dump(target_scaler, f)
        
    elif indx == 'ml_r' :
        # Save the features processor for the future tests and evaluations 
        preprocessor_path = "../results/ml_regression/"  
        os.makedirs(preprocessor_path, exist_ok=True)
        # Save target scaler separately
        with open(os.path.join(preprocessor_path, "target_scaler.pkl"), "wb") as f:
            pickle.dump(target_scaler, f)

    elif indx == 'dl_c':
        preprocessor_path = "../results/dl_classification/"  
        os.makedirs(preprocessor_path, exist_ok=True)

        with open( preprocessor_path+"label_map.pkl", "wb") as f:
            pickle.dump(label_map, f)

    elif indx == 'ml_c':
        preprocessor_path = "../results/ml_classification/"  
        os.makedirs(preprocessor_path, exist_ok=True)

        with open( preprocessor_path+"label_map.pkl", "wb") as f:
            pickle.dump(label_map, f)
    else :
        raise ValueError(f"Approche not supportable. (only ml and dl)")
    

    X = preprocessor.fit_transform(df[features])
    with open(os.path.join(preprocessor_path, "feature_preprocessor.pkl"), "wb") as f:
            pickle.dump(preprocessor, f)


    # If X is sparse (because of OneHotEncoder), convert to dense
    if hasattr(X, "toarray"):
        X = X.toarray()

    return X , Y

#-----------------------------------------------------------------------------------------------------------------

def split_reshape_normalize (X, Y, test_size=0.2, random_state=42):
     # Split data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )

        # Ensure X and Y are float64 before fitting
        X_train = np.asarray(X_train, dtype=np.float64)
        Y_train = np.asarray(Y_train, dtype=np.float64)
        X_test = np.asarray(X_test, dtype=np.float64)
        Y_test = np.asarray(Y_test, dtype=np.float64)


        X_train = torch.tensor(X_train, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.float32)        
        X_test = torch.tensor(X_test, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.float32)

        X_train = insure_none(X_train)
        Y_train = insure_none(Y_train)
        X_test = insure_none(X_test)
        Y_test = insure_none(Y_test)

        return X_train, X_test, Y_train, Y_test


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


def insure_none(x):
    if torch.isnan(x).any():
        print("Warning: NaN values detected in target y, replacing them with 0 to avoid loss and gradients calculus.(consider it as noise)")
        x = torch.nan_to_num(x, nan=0.0)
    return x 

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

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

# Define the model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    


    
