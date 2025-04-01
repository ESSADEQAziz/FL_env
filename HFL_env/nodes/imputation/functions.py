import pandas as pd
import numpy as np
import os
import math
import json

def evaluate_values(reference_json_path, local_parameters, aggegated_parameters,results_path):
 
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

        # we return the accuracy of the aggregated mean 
    return accuracy[1]

# This function aims to create some random missing data within a table based on a given missing_rate 
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
