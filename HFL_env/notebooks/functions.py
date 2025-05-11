import numpy as np
import torch
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


def check_data_structure(node_path):
    """
    Check the structure of the MIMIC-IV data
    """
    # Load the necessary tables
    chart_events = pd.read_csv(f'{node_path}/chartevents.csv')
    patients = pd.read_csv(f'{node_path}/patients.csv')
    
    print("Chart Events Info:")
    print(chart_events.info())
    print("\nChart Events Columns:")
    print(chart_events.columns.tolist())
    print("\nChart Events Sample:")
    print(chart_events.head())
    
    print("\nPatients Info:")
    print(patients.info())
    print("\nPatients Columns:")
    print(patients.columns.tolist())
    print("\nPatients Sample:")
    print(patients.head())

def map_mimic_features():
    """
    Map MIMIC-IV features to their categories and relationships
    """
    feature_mappings = {
        'vital_signs': {
            'chartevents': {
                'heart_rate': ['220045'],  # Heart Rate
                'blood_pressure_systolic': ['220050'],  # Blood Pressure
                'blood_pressure_diastolic': ['220052'],  # Blood Pressure
                'respiratory_rate': ['220210'],  # Respiratory Rate
                'spo2': ['220277']  # SpO2
            }
        },
        'lab_results': {
            'labevents': {
                'blood_glucose': ['50931'],  # Glucose
                'hemoglobin': ['51222'],  # Hemoglobin
                'wbc': ['51221'],  # White Blood Cells
                'platelet_count': ['51265'],  # Platelet Count
                'creatinine': ['50912']  # Creatinine
            }
        }
    }
    return feature_mappings

def prepare_mimic_data(node_path, feature_type='vital_signs'):
    """
    Prepare MIMIC-IV data for missing value analysis
    
    Args:
        node_path: Path to node data directory
        feature_type: 'vital_signs' or 'lab_results'
    """
    # Load the necessary tables
    print("\nLoading data tables...")
    try:
        if feature_type == 'vital_signs':
            events = pd.read_csv(f'{node_path}/chartevents.csv')
            print("Loaded chartevents.csv")
        else:  # lab_results
            events = pd.read_csv(f'{node_path}/labevents.csv')
            print("Loaded labevents.csv")
            
        patients = pd.read_csv(f'{node_path}/patients.csv')
        print("Loaded patients.csv")
        admissions = pd.read_csv(f'{node_path}/admissions.csv')
        print("Loaded admissions.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), [], []

    print("\n=== Events Table Info ===")
    print("\nColumns in events table:")
    print(events.columns.tolist())
    print("\nFirst few rows of events:")
    print(events.head())
    print("\nData types of events columns:")
    print(events.dtypes)
    print("\nNumber of unique values in each column:")
    for col in events.columns:
        print(f"{col}: {events[col].nunique()} unique values")
    
    # Get feature mappings
    feature_mappings = map_mimic_features()
    
    # Create a list to store DataFrames for each feature
    feature_dfs = []
    
    if feature_type == 'vital_signs':
        print("\nProcessing vital signs...")
        feature_dict = feature_mappings['vital_signs']['chartevents']
        value_col = 'valuenum'
    else:  # lab_results
        print("\nProcessing lab results...")
        feature_dict = feature_mappings['lab_results']['labevents']
        value_col = 'valuenum'
    
    for feature, itemids in feature_dict.items():
        print(f"\nProcessing {feature} with itemids {itemids}")
        
        # Convert itemids to integers
        try:
            itemids = [int(x) for x in itemids]
        except ValueError as e:
            print(f"Error converting itemids to int: {e}")
            continue
            
        # Filter events for the specific itemid
        try:
            feature_data = events[events['itemid'].isin(itemids)].copy()
            print(f"Found {len(feature_data)} rows for {feature}")
        except Exception as e:
            print(f"Error filtering data: {e}")
            continue
        
        if len(feature_data) > 0:
            print(f"Sample of data for {feature}:")
            print(feature_data.head())
            
            # Check if required columns exist
            required_cols = ['subject_id', 'hadm_id', 'charttime' if feature_type == 'vital_signs' else 'charttime', value_col]
            missing_cols = [col for col in required_cols if col not in feature_data.columns]
            if missing_cols:
                print(f"WARNING: Missing required columns: {missing_cols}")
                print(f"Available columns: {feature_data.columns.tolist()}")
                continue
            
            try:
                # Remove rows with NULL values in key columns
                feature_data = feature_data.dropna(subset=['subject_id', 'hadm_id', value_col])
                print(f"After removing NULL values: {len(feature_data)} rows")
                
                # Convert subject_id and hadm_id to appropriate types
                feature_data['subject_id'] = feature_data['subject_id'].astype('Int64')  # Using Int64 to handle NaN
                feature_data['hadm_id'] = feature_data['hadm_id'].astype('Int64')  # Using Int64 to handle NaN
                
                # Convert time to datetime
                time_col = 'charttime' if feature_type == 'vital_signs' else 'charttime'
                feature_data[time_col] = pd.to_datetime(feature_data[time_col])
                
                # Handle the value column
                if feature_type == 'lab_results':
                    # For lab results, convert to float to handle potential decimal values
                    feature_data[value_col] = pd.to_numeric(feature_data[value_col], errors='coerce')
                else:
                    # For vital signs, keep as is
                    feature_data[value_col] = pd.to_numeric(feature_data[value_col], errors='coerce')
                
                # Remove rows where the value is NULL after conversion
                feature_data = feature_data.dropna(subset=[value_col])
                print(f"After cleaning values: {len(feature_data)} rows")
                
                # Rename the value column to the feature name
                feature_data = feature_data.rename(columns={value_col: feature})
                
                # Select only necessary columns
                feature_data = feature_data[['subject_id', 'hadm_id', 'charttime' if feature_type == 'vital_signs' else 'charttime', feature]]
                
                if len(feature_data) > 0:
                    # Append to list only if we have data
                    feature_dfs.append(feature_data)
                    print(f"Added {feature} data to feature_dfs")
                else:
                    print(f"No valid data remaining for {feature}")
                    
            except Exception as e:
                print(f"Error processing {feature} data: {e}")
                continue
    
    if not feature_dfs:
        print("\nNo data found for any features!")
        return pd.DataFrame(), [], []
        
    # Merge all feature DataFrames
    print("\nMerging feature DataFrames...")
    try:
        data = feature_dfs[0]
        for df in feature_dfs[1:]:
            merge_cols = ['subject_id', 'hadm_id', 'charttime' if feature_type == 'vital_signs' else 'charttime']
            data = pd.merge(
                data,
                df,
                on=merge_cols,
                how='outer'
            )
    except Exception as e:
        print(f"Error merging feature DataFrames: {e}")
        return pd.DataFrame(), [], []
    
    print("\nMerging with patient demographics...")
    try:
        # Merge with patient demographics
        data = pd.merge(
            data,
            patients[['subject_id', 'gender', 'anchor_age']],
            on='subject_id',
            how='left'
        )
    except Exception as e:
        print(f"Error merging with demographics: {e}")
        return pd.DataFrame(), [], []
    
    # Select features for missingness and imputation
    if feature_type == 'vital_signs':
        features = list(feature_mappings['vital_signs']['chartevents'].keys())
    else:  # lab_results
        features = list(feature_mappings['lab_results']['labevents'].keys())
    
    # Select features for imputation
    imputation_features = features + ['gender', 'anchor_age']
    
    try:
        # Sort by subject_id, hadm_id, and time
        sort_cols = ['subject_id', 'hadm_id']
        sort_cols.append('charttime' if feature_type == 'vital_signs' else 'charttime')
        data = data.sort_values(sort_cols)
        
        # Reset index
        data = data.reset_index(drop=True)
        
        print("\nFinal data shape:", data.shape)
        print("\nFinal data columns:", data.columns.tolist())
        print("\nSample of final data:")
        print(data.head())
        
        return data, features, imputation_features
    except Exception as e:
        print(f"Error in final data processing: {e}")
        return pd.DataFrame(), [], []


def show_heatmap(data_path) :
    # 2. Load Dataset
    # Replace 'your_data.csv' with your actual file path
    df = pd.read_csv(data_path)

    # 3. Select Only Numeric Columns (regression only works with numeric values)
    numeric_df = df.select_dtypes(include=[np.number])

    # 4. Compute Correlation Matrix
    corr_matrix = numeric_df.corr()

    # 5. Create a Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap (Linear Feasibility)")
    plt.tight_layout()
    plt.show()

    # 6. Optional: Show top correlations with a specific target
    # Example: Correlation with 'target_column'
    target_column = 'your_target_column'
    if target_column in corr_matrix.columns:
        print(f"\nTop correlations with '{target_column}':")
        print(corr_matrix[target_column].sort_values(ascending=False))


def introduce_missing_values(csv_path, feature, target, missing_rate, pattern='MCAR'):
    """
    Introduce missing values in a specific feature with a given pattern and rate.
    
    Args:
        csv_path (str): Path to the CSV file containing the data
        feature (str): Name of the feature to introduce missing values
        target (str): Name of the target variable
        missing_rate (float): Rate of missing values to introduce (between 0 and 1)
        pattern (str): Pattern of missingness ('MCAR', 'MAR', or 'MNAR')
    
    Returns:
        tuple: (X_tensor, y_tensor) where:
            - X_tensor: Tensor of features with missing values
            - y_tensor: Tensor of target values
    """
    # Read the data
    df = pd.read_csv(csv_path)
    
    # Extract feature and target
    X = df[feature].values.reshape(-1, 1)
    y = df[target].values
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create a mask for missing values
    n_samples = len(X)
    n_missing = int(n_samples * missing_rate)
    
    if pattern == 'MCAR':
        # Missing Completely At Random
        # Randomly select indices to make missing
        missing_indices = np.random.choice(n_samples, n_missing, replace=False)
        X_scaled[missing_indices] = np.nan
        
    elif pattern == 'MAR':
        # Missing At Random
        # Missingness depends on other variables (here we'll use the target)
        # Sort by target values and create missing values in a structured way
        sorted_indices = np.argsort(y)
        missing_indices = sorted_indices[:n_missing]
        X_scaled[missing_indices] = np.nan

    elif pattern == 'MNAR':
        # Missing Not At Random
        # Missingness depends on the feature itself
        # Sort by feature values and create missing values in a structured way
        sorted_indices = np.argsort(X_scaled.flatten())
        missing_indices = sorted_indices[:n_missing]
        X_scaled[missing_indices] = np.nan
    
    else:
        raise ValueError("Pattern must be one of: 'MCAR', 'MAR', 'MNAR'")
    
    # Convert to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return X_tensor, y_tensor


