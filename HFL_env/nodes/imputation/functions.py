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

def calculate_statistics(file_path, feature):
    df = pd.read_csv(file_path)
    mean_value = df[feature].mean()
    median_value = df[feature].median()
    num_samples = len(df)
    return [mean_value, median_value],num_samples

#-----------------------------------------------------------------------------------------------------------------

def preprocess_node_data(csv_path, features, target, indx):
    df = pd.read_csv(csv_path)
    column_names = df.columns

    num_features = []
    cat_features = []

    for item in column_names:
        if item in features:
            if df[item].dtype in ['int64', 'float64']:
                num_features.append(item)
            elif df[item].dtype in ['object']:
                cat_features.append(item)

    # Handle missing values in numerical features
    for col in num_features:
        # Replace NaN values with column mean
        if df[col].isna().any():
            col_mean = df[col].mean()
            df[col].fillna(col_mean, inplace=True)
            # filled the numerical feature col with the mean col_mean

            
    # Handle missing values in categorical features
    for col in cat_features:
        if df[col].isna().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
            # filled the categorical feature col with the mode df[col].mode()[0]

    # Handle missing values in target
    if df[target].isna().any():
        if df[target].dtype in ['int64', 'float64']:
            target_mean = df[target].mean()
            df[target].fillna(target_mean, inplace=True)
            # Replaced NaN values in target with mean: target_mean
        elif df[target].dtype in ['object']:
            df[target].fillna(df[target].mode()[0], inplace=True)
            # Replaced NaN values in categorical target with the most frequent value 'mode'

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    if df[target].dtype in ['int64', 'float64']:
        # Preprocessor for target
        target_scaler = StandardScaler()
        Y = target_scaler.fit_transform(df[[target]])  # Keep as 2D array (n_samples, 1)
        label_map = None
    elif df[target].dtype in ['object']:
        target_f = pd.get_dummies(df[target])
        Y = pd.get_dummies(df[target]).values
        label_map = list(target_f.columns)
    else:
        raise ValueError("Failure processing target feature.")

    # Create appropriate directory structure based on indx
    if indx == 'dl_r':
        preprocessor_path = "../results/dl_regression/"
        os.makedirs(preprocessor_path, exist_ok=True)
        with open(os.path.join(preprocessor_path, "target_scaler.pkl"), "wb") as f:
            pickle.dump(target_scaler, f)
    elif indx == 'ml_r':
        preprocessor_path = "../results/ml_regression/"
        os.makedirs(preprocessor_path, exist_ok=True)
        with open(os.path.join(preprocessor_path, "target_scaler.pkl"), "wb") as f:
            pickle.dump(target_scaler, f)
    elif indx == 'dl_c':
        preprocessor_path = "../results/dl_classification/"
        os.makedirs(preprocessor_path, exist_ok=True)
        with open(preprocessor_path + "label_map.pkl", "wb") as f:
            pickle.dump(label_map, f)
    elif indx == 'ml_c':
        preprocessor_path = "../results/ml_classification/"
        os.makedirs(preprocessor_path, exist_ok=True)
        with open(preprocessor_path + "label_map.pkl", "wb") as f:
            pickle.dump(label_map, f)
    else:
        raise ValueError(f"Approach not supported. (only ml and dl)")

    X = preprocessor.fit_transform(df[features])
    with open(os.path.join(preprocessor_path, "feature_preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    # If X is sparse (because of OneHotEncoder), convert to dense
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Convert to torch tensors and handle any remaining NaN values with insure_none
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    
    # Handle any remaining NaN values that might have been introduced in processing
    X_tensor = insure_none(X_tensor, feature_type='numerical')
    Y_tensor = insure_none(Y_tensor, feature_type='numerical' if df[target].dtype in ['int64', 'float64'] else 'categorical', is_target=True)

    return X_tensor, Y_tensor

#-----------------------------------------------------------------------------------------------------------------

def split_reshape_normalize(X, Y, test_size=0.2, random_state=42):
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

def insure_none(x, feature_type='numerical', column_means=None, is_target=False):
    """
    Handle NaN values in tensor data by filling with appropriate values.
    
    Args:
        x: Tensor data potentially containing NaN values
        feature_type: Type of features ('numerical' or 'categorical')
        column_means: Precomputed column means for filling (if None, will compute from x)
        is_target: Whether this tensor is a target variable
        
    Returns:
        Tensor with NaN values replaced
    """
    if torch.isnan(x).any():
        if is_target:
            # For target variables, we might want special handling
            print("Warning: NaN values detected in target variable.")
            if feature_type == 'numerical':
                # For numerical targets, use mean imputation
                if column_means is None:
                    # Compute mean ignoring NaNs
                    column_means = torch.nanmean(x, dim=0, keepdim=True)
                
                print(f"Replacing NaN values in target with mean value: {column_means.item():.4f}")
                x = torch.nan_to_num(x, nan=column_means.item())
            else:
                # For categorical targets, this is more complex
                print("Warning: NaN values in categorical target may cause issues in model training.")
                # Fill with zeros, but this could be improved
                x = torch.nan_to_num(x, nan=0.0)
        else:
            # For feature variables
            if feature_type == 'numerical':
                # For numerical features, use column-wise mean imputation
                if column_means is None:
                    # Calculate mean for each feature, ignoring NaNs
                    column_means = torch.nanmean(x, dim=0, keepdim=True)
                
                # Create a mask of NaN values
                nan_mask = torch.isnan(x)
                
                # Fill NaN values with corresponding column means
                for col_idx in range(x.shape[1]):
                    col_mask = nan_mask[:, col_idx]
                    if col_mask.any():
                        x[col_mask, col_idx] = column_means[0, col_idx]
                
                print(f"Replaced NaN values in {torch.sum(nan_mask).item()} positions with column means.")
            
            elif feature_type == 'categorical':
                # For one-hot encoded categorical features
                # In one-hot encoding, NaN should translate to all zeros in that group
                nan_rows = torch.isnan(x).any(dim=1)
                if nan_rows.any():
                    print(f"Replaced NaN values in {torch.sum(nan_rows).item()} categorical instances with zeros.")
                    # Replace all NaNs with zeros (effectively making "unknown" category)
                    x = torch.nan_to_num(x, nan=0.0)
    
    return x


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

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    
def introduce_missing_values(data, feature_indices, missing_rate=0.2, pattern="MCAR"):
    """
    Artificially introduce missing values in the data.
    
    Args:
        data: Tensor or numpy array of shape [n_samples, n_features]
        feature_indices: List of indices where missing values should be introduced
        missing_rate: Proportion of values to set as missing (0.0-1.0)
        pattern: Missing pattern - "MCAR", "MAR", or "MNAR"
    
    Returns:
        data_with_missing: Same data with missing values (NaNs)
        missing_mask: Boolean mask where True indicates missing values
        original_values: Dictionary mapping (row, col) indices to original values
    """
    # Convert to numpy for processing
    is_tensor = torch.is_tensor(data)
    if is_tensor:
        device = data.device
        data_np = data.cpu().numpy()
    else:
        data_np = data.copy()
    n_samples, n_features = data_np.shape
    missing_mask = np.zeros_like(data_np, dtype=bool)
    original_values = {}
    
    for feat_idx in feature_indices:
        if feat_idx >= n_features:
            continue
            
        # Determine number of missing values for this feature
        n_missing = int(n_samples * missing_rate)
        
        if pattern == "MCAR":
            # Missing Completely At Random - uniform random selection
            missing_rows = np.random.choice(n_samples, n_missing, replace=False)
            
        elif pattern == "MAR":
            # Missing At Random - dependent on other features
            # Choose another feature as reference
            ref_feat = (feat_idx + 1) % n_features
            # Normalize to get probability distribution
            probs = np.abs(data_np[:, ref_feat])
            probs = probs / np.sum(probs)
            missing_rows = np.random.choice(n_samples, n_missing, replace=False, p=probs)

        elif pattern == "MNAR":
            # Missing Not At Random - dependent on its own value
            # Higher values more likely to be missing
            probs = np.abs(data_np[:, feat_idx])
            probs = probs / np.sum(probs)
            missing_rows = np.random.choice(n_samples, n_missing, replace=False, p=probs)
        
        # Store original values and set to NaN
        for row in missing_rows:
            original_values[(row, feat_idx)] = float(data_np[row, feat_idx])
            missing_mask[row, feat_idx] = True
            data_np[row, feat_idx] = np.nan
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        data_with_missing = torch.tensor(data_np, dtype=torch.float32, device=device)
        missing_mask = torch.tensor(missing_mask, dtype=torch.bool, device=device)
    else:
        data_with_missing = data_np
    
    return data_with_missing, missing_mask, original_values


def evaluate_imputation_metrics(original_values, imputed_values):
    """
    Calculate metrics to evaluate imputation quality
    
    Args:
        original_values: Dict mapping (row, col) to original values
        imputed_values: Array containing imputed values at the same positions
    
    Returns:
        Dictionary of metrics
    """
    if not original_values:
        return {"error": "No values to impute"}
        
    # Extract values for comparison
    orig = []
    imp = []
    
    for (row, col), true_val in original_values.items():
        orig.append(true_val)
        imp.append(imputed_values[row, col])
    
    orig = np.array(orig)
    imp = np.array(imp)
    
    # Calculate metrics
    mae = np.mean(np.abs(orig - imp))
    rmse = np.sqrt(np.mean((orig - imp) ** 2))

    # Mean Absolute Percentage Error (handle zeros)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((orig - imp) / orig)) * 100
        mape = np.nan_to_num(mape, nan=0)
    
    # Correlation between original and imputed values
    correlation = np.corrcoef(orig, imp)[0, 1] if len(orig) > 1 else 0
    
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "correlation": float(correlation)
    }

