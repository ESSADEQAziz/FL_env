import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from IPython.display import display, Markdown

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
    with proper handling of NaN values.
    
    Args:
        df: DataFrame containing the data
        target_col: Target column to calculate MI against
    """
    # Get numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target_col not in numeric_df.columns:
        print(f"Error: {target_col} is not a numeric column in the dataframe")
        return
    
    # Get target variable
    y = numeric_df[target_col]
    
    # Drop rows with NaN in target variable
    if y.isna().any():
        print(f"Warning: Target column '{target_col}' contains {y.isna().sum()} NaN values. These rows will be dropped.")
        valid_indices = ~y.isna()
        y = y[valid_indices]
    else:
        valid_indices = pd.Series([True] * len(df), index=df.index)
    
    # Get feature variables, excluding target
    X = numeric_df.drop(columns=[target_col])
    
    # Handle NaN values in features
    if X.isna().any().any():
        print("Warning: Some feature columns contain NaN values:")
        nan_counts = X.isna().sum()
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  - {col}: {count} NaN values ({count/len(X):.2%})")
        
        # Use only rows where both X and y are valid
        X = X.loc[valid_indices]
        
        # Fill remaining NaNs with median for each feature
        for col in X.columns:
            if X[col].isna().any():
                median_value = X[col].median()
                X[col] = X[col].fillna(median_value)
                print(f"  - Filled NaNs in '{col}' with median value: {median_value}")
    else:
        # Use only rows where target is valid
        X = X.loc[valid_indices]
    
    # Verify we have data to work with
    if len(X) == 0 or len(y) == 0:
        print("Error: After handling NaN values, no data remains for analysis.")
        return
    
    print(f"Computing mutual information on {len(X)} samples after handling NaN values.")
    
    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y)
    
    # Create DataFrame with scores
    mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
    mi_df = mi_df.sort_values('MI Score', ascending=False)
    
    print("\n=== Mutual Information with target:", target_col, "===")
    print(mi_df)
    
    # Plot MI scores
    plt.figure(figsize=(10, 6))
    sns.barplot(x='MI Score', y='Feature', data=mi_df)
    plt.title(f'Mutual Information Scores (target: {target_col})')
    plt.tight_layout()
    plt.show()
    
    return mi_df
 
def introduce_missingness(df, feature, missing_rate, pattern='MCAR', target=None, seed=42):
    """
    Introduce missing values in a specific feature of a DataFrame with a given pattern and rate,
    while saving the original values for later evaluation.
    
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
            - ground_truth: DataFrame with original values and missingness mask
    """
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Check if feature exists in DataFrame
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame columns: {df.columns.tolist()}")
    
    # Check if we need target for MAR pattern
    if pattern == 'MAR' and (target is None or target not in df.columns):
        raise ValueError(f"Target variable is required for MAR pattern. Target '{target}' not found.")
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Extract feature values
    X = df_copy[feature].values
    
    # Create a ground truth DataFrame to store original values
    ground_truth = pd.DataFrame()
    ground_truth['original_values'] = df_copy[feature].copy()
    
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
    valid_indices = np.where(~df_copy[feature].isna())[0]
    
    if len(valid_indices) == 0:
        raise ValueError(f"Feature '{feature}' has no valid values. Cannot introduce missingness.")
    
    # Calculate number of values to make missing
    n_to_make_missing = int(n_samples * missing_rate - existing_nan_count)
    
    if n_to_make_missing <= 0:
        print(f"Feature already has {existing_nan_rate:.2%} missing values, "
              f"which is greater than or equal to the requested {missing_rate:.2%}.")
        # Return original DataFrame and ground truth
        ground_truth['is_missing'] = original_mask
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
        target_values = df_copy[target].values
        
        # Create pairs of valid indices and their corresponding target values
        valid_with_target = [(i, target_values[i]) for i in valid_indices]
        
        # Sort by target values and select indices with lowest/highest target values
        sorted_pairs = sorted(valid_with_target, key=lambda x: x[1])
        missing_indices = [pair[0] for pair in sorted_pairs[:n_to_make_missing]]
    
    elif pattern == 'MNAR':
        # Missing Not At Random - depends on feature value itself
        feature_values = df_copy[feature].values
        
        # Create pairs of valid indices and their corresponding feature values
        valid_with_feature = [(i, feature_values[i]) for i in valid_indices]
        
        # Sort by feature values and select indices with lowest/highest feature values
        sorted_pairs = sorted(valid_with_feature, key=lambda x: x[1])
        missing_indices = [pair[0] for pair in sorted_pairs[:n_to_make_missing]]
    
    else:
        raise ValueError("Pattern must be one of: 'MCAR', 'MAR', 'MNAR'")
    
    # Create missingness mask for ground truth
    missingness_mask = np.zeros(n_samples, dtype=bool)
    missingness_mask[missing_indices] = True
    
    # Add existing missing values to the mask
    missingness_mask = missingness_mask | original_mask.values
    
    # Store missingness mask in ground truth
    ground_truth['is_missing'] = missingness_mask
    
    # Introduce missing values in the copy
    df_copy.loc[missing_indices, feature] = np.nan
    
    # Calculate final missing rate for verification
    final_missing_count = df_copy[feature].isna().sum()
    final_missing_rate = final_missing_count / n_samples
    
    print(f"Missingness pattern: {pattern}")
    print(f"Original missing rate: {existing_nan_rate:.2%}")
    print(f"Added {n_to_make_missing} missing values")
    print(f"Final missing rate: {final_missing_rate:.2%}")
    
    # Add useful information to ground truth
    ground_truth['artificially_masked'] = missingness_mask & ~original_mask.values
    
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
