import logging
import pandas as pd
import numpy as np
import json
import os
import flwr as fl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/nodes.log"),
    ]
)
logger = logging.getLogger("node")
NODE_ID = os.environ.get("NODE_ID", "1")
logger.info(f"Starting node {NODE_ID}")


# This function aims to create some random missing data within a table based on a given missing_rate 
def create_missing_values(csv_path: str, feature_name: str, missing_rate: float, 
                          output_dir: str = "./missing_data") -> str:
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
   
def get_client(feature,path):
    """Get the appropriate client based on the imputation strategy."""
    strategy = os.environ.get("IMPUTATION_STRATEGY", "statistical")
    
    if strategy == "statistical":
        from imputation.statistical import NodeClient
        return NodeClient(feature,path)
    elif strategy == "machine_learning":
        from imputation.machine_learning import NodeClient
        return NodeClient()
    elif strategy == "deep_learning":
        from imputation.deep_learning import NodeClient
        return NodeClient()
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

if __name__ == "__main__":
    path_to_missing_data = create_missing_values('../data/chartevents.csv','valuenum',0.1)
    client = get_client('valuenum',path_to_missing_data)
    fl.client.start_numpy_client(
        server_address="central_server:5000",
        client=client
    )