import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.compose import ColumnTransformer
from IPython.display import display
import torch
import pickle
import os


def analyze_dataframe(df=None, csv_path=None):
    """
    Display basic information about a dataframe with simple table formatting.
    
    Args:
        df: pandas DataFrame (optional if csv_path is provided)
        csv_path: Path to CSV file (optional if df is provided)
    
    Returns:
        The dataframe being analyzed
    """
    # Load data if dataframe not provided
    if df is None:
        if csv_path is None:
            print("Error: Either df or csv_path must be provided.")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            print(f"Dataset loaded from: {csv_path}")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None
    
    # Basic dataframe information
    print("\n## DATAFRAME OVERVIEW")
    print("-" * 40)
    
    overview_data = {
        "Metric": ["Rows", "Columns", "Total Elements", "Memory Usage (MB)"],
        "Value": [
            df.shape[0], 
            df.shape[1], 
            df.size, 
            f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f}"
        ]
    }
    overview_df = pd.DataFrame(overview_data)
    display(overview_df)
    
    # Column information with missingness stats
    print("\n## COLUMN INFORMATION")
    print("-" * 40)
    
    column_info = []
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_rate = missing_count / len(df)
        
        column_info.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Unique Values": df[col].nunique(),
            "Unique %": f"{(df[col].nunique() / len(df)) * 100:.1f}%",
            "Missing Count": missing_count,
            "Missing %": f"{missing_rate:.2%}"
        })
    
    info_df = pd.DataFrame(column_info)
    display(info_df)
    
    # Sample data
    print("\n## SAMPLE DATA (First 5 rows)")
    print("-" * 40)
    display(df.head())
    
    # Missingness summary
    print("\n## MISSINGNESS SUMMARY")
    print("-" * 40)
    
    # Count features with missing values
    features_with_missing = info_df[info_df["Missing Count"] > 0]["Column"].tolist()
    if features_with_missing:
        print(f"Features with missing values: {len(features_with_missing)} out of {len(df.columns)}")
        print(f"Features with missing values: {', '.join(features_with_missing)}")
        
        # Sort by missing percentage in descending order
        missing_sorted = info_df.sort_values("Missing Count", ascending=False)
        print("\nTop features with highest missingness:")
        
        top_missing = missing_sorted[["Column", "Missing Count", "Missing %"]].head(10)
        display(top_missing)
    else:
        print("No missing values found in the dataset.")
    
    return df

def calculate_spearman_correlation(df, target_col=None):
    """
    Calculate and print Spearman rank correlation for all numeric columns or against a specific target,
    with proper handling of NaN values.
    
    Args:
        df: DataFrame containing the data
        target_col: Optional target column to correlate against
    """
    # Get numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Check for and report NaN values
    total_rows = len(numeric_df)
    nan_counts = numeric_df.isna().sum()
    columns_with_nans = nan_counts[nan_counts > 0]
    
    if len(columns_with_nans) > 0:
        print("\n=== NaN Values in Numeric Columns ===")
        for col, count in columns_with_nans.items():
            print(f"  - {col}: {count} NaN values ({count/total_rows:.2%})")
        
        print("\nNote: NaN values will be automatically excluded from pairwise correlation calculations.")
    
    if target_col:
        # Correlation against specific target
        if target_col not in numeric_df.columns:
            print(f"Error: {target_col} is not a numeric column in the dataframe")
            return
        
        # Check NaNs in target column
        target_nans = numeric_df[target_col].isna().sum()
        if target_nans > 0:
            print(f"\nWarning: Target column '{target_col}' has {target_nans} NaN values ({target_nans/total_rows:.2%}).")
            print("These rows will be excluded when calculating correlations.")
        
        # Calculate correlations (pandas automatically handles NaNs with pairwise deletion)
        corr = numeric_df.corrwith(numeric_df[target_col], method='spearman').sort_values(ascending=False)
        
        # Drop NaN correlation results
        if corr.isna().any():
            print("\nWarning: Some features have NaN correlation with the target.")
            print("This typically happens when a feature has no valid pairs with the target or contains constant values.")
            print("These features will be removed from the results:")
            for col in corr[corr.isna()].index:
                print(f"  - {col}")
            
            corr = corr.dropna()
        
        # Create DataFrame with results
        corr_df = pd.DataFrame(corr, columns=['Spearman Correlation'])
        
        # Report number of observations used
        valid_pairs = {}
        for col in corr_df.index:
            if col != target_col:
                valid_count = numeric_df[[col, target_col]].dropna().shape[0]
                valid_pairs[col] = valid_count
        
        corr_df['Valid Observations'] = pd.Series(valid_pairs)
        corr_df['% of Data Used'] = (corr_df['Valid Observations'] / total_rows * 100).round(2).astype(str) + '%'
        
        print("\n=== Spearman Rank Correlation with target:", target_col, "===")
        print(corr_df)
        
        # Plot the correlation values as a bar chart (only for correlations, not metadata)
        plt.figure(figsize=(10, 6))
        plot_data = corr_df.copy()
        # Sort by absolute correlation value for better visualization
        plot_data = plot_data.sort_values('Spearman Correlation', key=abs, ascending=False)
        
        sns.barplot(x=plot_data.index, y='Spearman Correlation', data=plot_data)
        plt.xticks(rotation=90)
        plt.title(f'Spearman Correlation with {target_col}')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Add a horizontal line at y=0
        plt.tight_layout()
        plt.show()
        
        return corr_df
    
    else:
        # Full correlation matrix (pandas handles NaNs with pairwise deletion)
        corr = numeric_df.corr(method='spearman')
        
        # Create a counts matrix to show number of observations used for each pair
        obs_counts = pd.DataFrame(index=corr.index, columns=corr.columns)
        for i in corr.index:
            for j in corr.columns:
                obs_counts.loc[i, j] = numeric_df[[i, j]].dropna().shape[0]
        
        print("\n=== Spearman Rank Correlation Matrix ===")
        print(corr)
        
        print("\n=== Number of observations used for each correlation pair ===")
        print(obs_counts)
        
        # Calculate percentage of data used
        pct_used = (obs_counts / total_rows * 100).round(2)
        print("\n=== Percentage of data used for each correlation pair ===")
        print(pct_used)
        
        # Check for any low data usage pairs
        low_data_pairs = []
        threshold = 0.7  # Flag pairs using less than 70% of the data
        for i in pct_used.index:
            for j in pct_used.columns:
                if i != j and pct_used.loc[i, j] < threshold * 100:
                    low_data_pairs.append((i, j, pct_used.loc[i, j]))
        
        if low_data_pairs:
            print("\nWarning: Some correlation pairs use a low percentage of the data:")
            for i, j, pct in sorted(low_data_pairs, key=lambda x: x[2]):
                print(f"  - {i} vs {j}: {pct:.1f}% of data used")
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, 
                    center=0, square=True, linewidths=.5, annot=True, fmt=".2f")
        plt.title("Spearman Correlation Matrix")
        plt.tight_layout()
        plt.show()
        
        return corr

def calculate_mutual_information(df, target_col):
    """
    Calculate and print mutual information between features and target,
    with pairwise deletion of NaN values for each feature-target pair.
    
    Args:
        df: DataFrame containing the data
        target_col: Target column to calculate MI against
    """
    # Get numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target_col not in numeric_df.columns:
        print(f"Error: {target_col} is not a numeric column in the dataframe")
        return
    
    # Check for and report NaN values
    total_rows = len(numeric_df)
    nan_counts = numeric_df.isna().sum()
    columns_with_nans = nan_counts[nan_counts > 0]
    
    if len(columns_with_nans) > 0:
        print("\n=== NaN Values in Numeric Columns ===")
        for col, count in columns_with_nans.items():
            print(f"  - {col}: {count} NaN values ({count/total_rows:.2%})")
        
        print("\nNote: NaN values will be excluded on a pairwise basis for each feature-target pair.")
    
    # Check NaNs in target column
    target_nans = numeric_df[target_col].isna().sum()
    if target_nans > 0:
        print(f"\nWarning: Target column '{target_col}' has {target_nans} NaN values ({target_nans/total_rows:.2%}).")
        print("Rows with NaN in the target will be excluded from all calculations.")
    
    # Get features excluding target
    features = [col for col in numeric_df.columns if col != target_col]
    
    # Calculate MI for each feature with pairwise deletion of NaNs
    mi_scores = []
    valid_pairs = {}
    
    for feature in features:
        # Create a clean subset with this feature and the target
        pair_df = numeric_df[[feature, target_col]].dropna()
        valid_count = len(pair_df)
        valid_pairs[feature] = valid_count
        
        # Skip features with too few valid pairs
        if valid_count < 5:
            print(f"Warning: Feature '{feature}' has only {valid_count} valid pairs with target. Skipping MI calculation.")
            mi_scores.append((feature, np.nan))
            continue
        
        # Calculate MI for this pair
        try:
            X = pair_df[[feature]].values  # sklearn expects 2D array
            y = pair_df[target_col].values
            mi = mutual_info_regression(X, y)[0]  # [0] because we get a 1-element array
            mi_scores.append((feature, mi))
        except Exception as e:
            print(f"Error calculating MI for feature '{feature}': {e}")
            mi_scores.append((feature, np.nan))
    
    # Create DataFrame with scores
    mi_df = pd.DataFrame(mi_scores, columns=['Feature', 'MI Score'])
    
    # Add information about valid observations
    valid_obs_series = pd.Series(valid_pairs)
    # Make sure the indices match
    mi_df['Valid Observations'] = mi_df['Feature'].map(valid_obs_series)
    mi_df['% of Data Used'] = (mi_df['Valid Observations'] / total_rows * 100).round(2).astype(str) + '%'
    
    # Sort by MI Score, handling NaN values
    mi_df = mi_df.sort_values('MI Score', ascending=False, na_position='last')
    
    # Handle cases where some features have NaN MI scores
    if mi_df['MI Score'].isna().any():
        print("\nWarning: Some features have NaN Mutual Information scores with the target.")
        print("This typically happens when a feature has too few valid pairs with the target.")
        print("These features will be shown at the end of the results.")
    
    print("\n=== Mutual Information with target:", target_col, "===")
    print(mi_df)
    
    # Plot MI scores (excluding NaN values)
    if len(mi_df.dropna(subset=['MI Score'])) > 0:
        plt.figure(figsize=(10, 6))
        plot_data = mi_df.dropna(subset=['MI Score']).copy()
        sns.barplot(x='MI Score', y='Feature', data=plot_data)
        plt.title(f'Mutual Information Scores (target: {target_col})')
        plt.tight_layout()
        plt.show()
    else:
        print("No valid MI scores to plot.")
    
    # Check for any low data usage pairs
    low_data_pairs = mi_df[mi_df['Valid Observations'] < 0.7 * total_rows]
    if len(low_data_pairs) > 0:
        print("\nWarning: Some feature-target pairs use a low percentage of the data:")
        for _, row in low_data_pairs.iterrows():
            print(f"  - {row['Feature']}: {row['% of Data Used']} of data used")
    
    return mi_df

def introduce_missingness(df, feature, missing_rate, pattern='MCAR', target=None, seed=42):
    """
    Introduce missing values in a specific feature of a DataFrame with a given pattern and rate,
    while saving only the original values paired with charttime.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        feature (str): Name of the feature to introduce missing values
        missing_rate (float): Rate of missing values to introduce (between 0 and 1)
        pattern (str): Pattern of missingness ('MCAR', 'MAR', or 'MNAR')
        target (str, optional): Name of the target variable (needed for MAR pattern)
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (df_with_missing, ground_truth) where:
            - df_with_missing: DataFrame with missing values introduced
            - ground_truth: DataFrame with only original values and charttime
    """
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Check if feature exists in DataFrame
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame columns: {df.columns.tolist()}")
    
    # Check if charttime exists
    if 'charttime' not in df.columns:
        raise ValueError("Column 'charttime' not found in DataFrame.")
    
    # Check if we need target for MAR pattern
    if pattern == 'MAR' and (target is None or target not in df.columns):
        raise ValueError(f"Target variable is required for MAR pattern. Target '{target}' not found.")
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Create a ground truth DataFrame with only original values and charttime
    ground_truth = pd.DataFrame(index=df_copy.index)
    ground_truth['original_values'] = df_copy[feature].copy()
    ground_truth['charttime'] = df_copy['charttime'].copy()
    
    # Check for existing NaN values
    original_mask = df_copy[feature].isna()
    existing_nan_count = original_mask.sum()
    n_samples = len(df_copy)
    existing_nan_rate = existing_nan_count / n_samples
    
    # Calculate how many values to make missing
    if existing_nan_rate > 0:
        print(f"Warning: Feature '{feature}' already has {existing_nan_rate:.2%} missing values.")
        print("Using only rows with existing values to introduce new missingness.")
    
    # Focus only on non-missing values for introducing missingness
    valid_mask = ~df_copy[feature].isna()
    valid_indices = df_copy.index[valid_mask].tolist()
    
    if len(valid_indices) == 0:
        raise ValueError(f"Feature '{feature}' has no valid values. Cannot introduce missingness.")
    
    # Calculate number of values to make missing
    n_to_make_missing = int(n_samples * missing_rate - existing_nan_count)
    
    if n_to_make_missing <= 0:
        print(f"Feature already has {existing_nan_rate:.2%} missing values, "
              f"which is greater than or equal to the requested {missing_rate:.2%}.")
        return df_copy, ground_truth
    
    if n_to_make_missing > len(valid_indices):
        print(f"Warning: Cannot introduce {n_to_make_missing} missing values as only "
              f"{len(valid_indices)} non-missing values are available.")
        n_to_make_missing = len(valid_indices)
    
    # Select indices to make missing based on pattern
    missing_indices = []
    
    if pattern == 'MCAR':
        # Missing Completely At Random
        missing_indices = np.random.choice(valid_indices, n_to_make_missing, replace=False)
    
    elif pattern == 'MAR':
        # Missing At Random - depends on target variable
        # Create a DataFrame with valid indices and target values
        target_df = pd.DataFrame({
            'index': valid_indices,
            'target_value': df_copy.loc[valid_indices, target].values
        })
        
        # Sort by target values and select indices with lowest target values
        target_df = target_df.sort_values('target_value')
        missing_indices = target_df['index'].head(n_to_make_missing).tolist()
    
    elif pattern == 'MNAR':
        # Missing Not At Random - depends on feature value itself
        # Create a DataFrame with valid indices and feature values
        feature_df = pd.DataFrame({
            'index': valid_indices,
            'feature_value': df_copy.loc[valid_indices, feature].values
        })
        
        # Sort by feature values and select indices with lowest feature values
        feature_df = feature_df.sort_values('feature_value')
        missing_indices = feature_df['index'].head(n_to_make_missing).tolist()
    
    else:
        raise ValueError("Pattern must be one of: 'MCAR', 'MAR', 'MNAR'")
    
    # Introduce missing values in the copy
    df_copy.loc[missing_indices, feature] = np.nan
    
    # Calculate final missing rate for verification
    final_missing_count = df_copy[feature].isna().sum()
    final_missing_rate = final_missing_count / n_samples
    
    print(f"Missingness pattern: {pattern}")
    print(f"Original missing rate: {existing_nan_rate:.2%}")
    print(f"Added {n_to_make_missing} missing values")
    print(f"Final missing rate: {final_missing_rate:.2%}")
    
    return df_copy, ground_truth

def prepare_clean_dataset(df, feature1, feature2):
    """
    Create a new DataFrame with only rows that have non-missing values for both specified features.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        feature1 (str): First feature name to check for missing values
        feature2 (str): Second feature name to check for missing values
        
    Returns:
        pandas.DataFrame: Clean dataset with no missing values in both specified features
    """
    # Filter rows with non-missing values in both features
    clean_df = df.dropna(subset=[feature1, feature2])
    
    # Report statistics
    original_count = len(df)
    clean_count = len(clean_df)
    
    print(f"Original dataset: {original_count} rows")
    print(f"Clean dataset: {clean_count} rows ({clean_count/original_count:.2%} of original)")
    print(f"\nMissing values summary:")
    print(f"{feature1}: {df[feature1].isna().sum()} missing values")
    print(f"{feature2}: {df[feature2].isna().sum()} missing values")
    print(f"Rows missing either feature: {original_count - clean_count}")
    
    return clean_df

def convert_feature_to_float(df, feature_name):
    """
    Convert a feature to float type if it contains string values.
    
    Args:
        df: pandas DataFrame
        feature_name: name of the feature to convert
        
    Returns:
        DataFrame with the feature converted to float if possible
    """
    # Make a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Check if the feature exists
    if feature_name not in df_copy.columns:
        print(f"Error: Feature '{feature_name}' not found in the DataFrame")
        return df_copy
    
    # Check current dtype
    current_dtype = df_copy[feature_name].dtype
    print(f"Current dtype of '{feature_name}': {current_dtype}")
    
    # If already numeric, return the DataFrame
    if np.issubdtype(current_dtype, np.number):
        print(f"Feature '{feature_name}' is already numeric type ({current_dtype})")
        return df_copy
    
    # Try to convert to float
    try:
        # Handle common issues in string representations of numbers
        if df_copy[feature_name].dtype == 'object':
            # Replace common string patterns
            df_copy[feature_name] = df_copy[feature_name].astype(str)
            df_copy[feature_name] = df_copy[feature_name].str.replace(',', '.')  # European decimal separator
            df_copy[feature_name] = df_copy[feature_name].str.replace(' ', '')   # Remove spaces
            df_copy[feature_name] = df_copy[feature_name].str.replace('$', '')   # Remove dollar signs
            df_copy[feature_name] = df_copy[feature_name].str.replace('%', '')   # Remove percent signs
            
            # Replace empty strings with NaN
            df_copy[feature_name] = df_copy[feature_name].replace('', np.nan)
        
        # Convert to float
        df_copy[feature_name] = df_copy[feature_name].astype(float)
        print(f"Successfully converted '{feature_name}' to float type")
        
        # Report missing values after conversion
        missing_count = df_copy[feature_name].isna().sum()
        print(f"Missing values after conversion: {missing_count} ({missing_count/len(df_copy):.2%})")
        
    except Exception as e:
        print(f"Error converting '{feature_name}' to float: {e}")
        print("Examples of problematic values:")
        # Show some examples of values that couldn't be converted
        non_convertible = df_copy[~df_copy[feature_name].astype(str).str.replace(',', '.').str.replace(' ', '').str.replace('$', '').str.replace('%', '').str.match(r'^[-+]?[0-9]*\.?[0-9]+$')]
        if len(non_convertible) > 0:
            print(non_convertible[feature_name].head())
    
    return df_copy

# Extended convert_feature_to_float to handle multiple features
def convert_features_to_float(df, feature_names):
    """
    Convert multiple features to float type if they contain string values.
    
    Args:
        df: pandas DataFrame
        feature_names: list of feature names to convert
        
    Returns:
        DataFrame with features converted to float if possible
    """
    df_result = df.copy()
    
    for feature in feature_names:
        df_result = convert_feature_to_float(df_result, feature)
        print("-" * 50)  # Separator between features
    
    return df_result

def load_model_and_preprocessors(model_path, preprocessor_dir, approach):
    """
    Load the saved model and all necessary preprocessors.
    
    Args:
        model_path: Path to the saved model (.pth file)
        preprocessor_dir: Directory containing preprocessors
        approach: 'dl_r', 'ml_r', 'dl_c', or 'ml_c'
    
    Returns:
        model: Loaded PyTorch model
        feature_preprocessor: Loaded feature preprocessor
        target_transformer: Target scaler or label map
    """
    # Load the model
    model = torch.load(model_path)
    model.eval()  # Set to evaluation mode
    
    # Load the feature preprocessor
    with open(os.path.join(preprocessor_dir, "feature_preprocessor.pkl"), "rb") as f:
        feature_preprocessor = pickle.load(f)
    
    # Load target transformer based on approach
    if approach in ['dl_r', 'ml_r']:
        # Regression - load target scaler
        with open(os.path.join(preprocessor_dir, "target_scaler.pkl"), "rb") as f:
            target_transformer = pickle.load(f)
    else:
        # Classification - load label map
        with open(os.path.join(preprocessor_dir, "label_map.pkl"), "rb") as f:
            target_transformer = pickle.load(f)
    
    return model, feature_preprocessor, target_transformer


def predict(model, df, features, feature_preprocessor, target_transformer, approach):
    """
    Make predictions using a saved model with proper preprocessing.
    
    Args:
        model: Loaded PyTorch model
        df: DataFrame with input data
        features: List of feature names
        feature_preprocessor: Loaded feature preprocessor
        target_transformer: Target scaler or label map
        approach: 'dl_r', 'ml_r', 'dl_c', or 'ml_c'
    
    Returns:
        predictions: Processed predictions in original scale
    """
    # Preprocess features using saved preprocessor
    X = feature_preprocessor.transform(df[features])
    
    # Convert to dense if needed
    if hasattr(X, "toarray"):
        X = X.toarray()
    
    # Convert to torch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Handle NaN values if any
    if torch.isnan(X_tensor).any():
        print("Warning: NaN values detected in input data. Handling NaNs...")
        # Get column information from preprocessor to determine numerical vs categorical
        num_features = []
        cat_features = []
        # Extract information from the ColumnTransformer
        for name, transformer, columns in feature_preprocessor.transformers_:
            if name == 'num':
                num_features_count = len(columns)
                num_idx = (0, num_features_count)
            elif name == 'cat':
                cat_features_count = X.shape[1] - num_features_count
                cat_idx = (num_features_count, X.shape[1])
        
        # Handle NaNs by feature type
        feature_types = {}
        if 'num_idx' in locals():
            feature_types[num_idx] = 'numerical'
        if 'cat_idx' in locals():
            feature_types[cat_idx] = 'categorical'
        
        # Apply NaN handling for each feature type
        for range_or_idx, feat_type in feature_types.items():
            if isinstance(range_or_idx, tuple):
                # Range of columns
                start, end = range_or_idx
                X_subset = X_tensor[:, start:end]
                if feat_type == 'numerical':
                    # For numerical features, use column-wise mean imputation
                    column_means = torch.nanmean(X_subset, dim=0, keepdim=True)
                    nan_mask = torch.isnan(X_subset)
                    for col_idx in range(X_subset.shape[1]):
                        col_mask = nan_mask[:, col_idx]
                        if col_mask.any():
                            X_subset[col_mask, col_idx] = column_means[0, col_idx]
                else:
                    # For categorical features, fill with zeros
                    X_subset = torch.nan_to_num(X_subset, nan=0.0)
                
                # Put back into X_tensor
                X_tensor[:, start:end] = X_subset
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_tensor)
    
    # Convert to numpy
    predictions_np = predictions.numpy()
    # Process predictions based on approach
    if approach in ['dl_r', 'ml_r']:
        # Regression - inverse transform to get original scale
        predictions_original = target_transformer.inverse_transform(predictions_np)
        return predictions_original
    else:
        # Classification - convert to class labels
        if predictions_np.shape[1] > 1:
            # Multi-class: get class with highest probability
            pred_classes = np.argmax(predictions_np, axis=1)
            # Map indices to labels
            return [target_transformer[i] for i in pred_classes]
        else:
            # Binary classification
            pred_classes = (predictions_np > 0.5).astype(int)
            # Map 0/1 to actual class labels
            return [target_transformer[i[0]] for i in pred_classes]


















