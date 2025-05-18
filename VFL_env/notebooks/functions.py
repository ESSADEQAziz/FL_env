import numpy as np
import pandas as pd
import os
import torch
import json
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
import pickle
import warnings
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import List
from sklearn.feature_selection import mutual_info_regression
from IPython.display import display
from torch.serialization import add_safe_globals


def merge_csvs_on_feature(csv_paths: List[str], merge_on: str, how: str = 'inner') -> pd.DataFrame:
    """
    Merges multiple CSV files on a specified feature column.

    Parameters:
        csv_paths (List[str]): List of CSV file paths to merge.
        merge_on (str): The column name to merge on.
        how (str): Type of merge to perform: 'inner', 'outer', 'left', or 'right'. Default is 'inner'.

    Returns:
        pd.DataFrame: The merged DataFrame with duplicate columns (by name) removed.
    """
    if not csv_paths:
        raise ValueError("No CSV paths provided.")

    # Read the first CSV
    merged_df = pd.read_csv(csv_paths[0])

    for path in csv_paths[1:]:
        df = pd.read_csv(path)
        
        # Avoid duplicate columns: keep only columns that are not already in merged_df (except merge_on)
        duplicate_columns = [col for col in df.columns if col in merged_df.columns and col != merge_on]
        df = df.drop(columns=duplicate_columns)

        merged_df = pd.merge(merged_df, df, on=merge_on, how=how)

    return merged_df

def analyze_feature(df, feature):
    """
    Analyze a single feature in a dataframe, providing comprehensive statistics and visualizations.
    
    Args:
        df: pandas DataFrame
        feature: name of the column to analyze
    """
    # Check if feature exists
    if feature not in df.columns:
        print(f"Error: Feature '{feature}' not found in DataFrame columns.")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Get the series
    series = df[feature]
    
    print(f"\n{'='*50}")
    print(f"ANALYSIS OF FEATURE: {feature}")
    print(f"{'='*50}")
    
    # Basic information
    print("\n--- Basic Information ---")
    print(f"Data type: {series.dtype}")
    null_count = series.isna().sum()
    null_percent = (null_count / len(series)) * 100
    print(f"Missing values: {null_count} ({null_percent:.2f}%)")
    print(f"Unique values: {series.nunique()} ({series.nunique()/len(series)*100:.2f}% of total)")
    
    # For categorical features
    if series.dtype == 'object' or series.dtype.name == 'category' or series.nunique() < 10:
        print("\n--- Categorical Analysis ---")
        value_counts = series.value_counts()
        value_percent = series.value_counts(normalize=True) * 100
        
        # Create a DataFrame with counts and percentages
        cat_df = pd.DataFrame({
            'Count': value_counts,
            'Percentage': value_percent
        })
        print(cat_df)
        
        # Plot bar chart for categorical data
        plt.figure(figsize=(10, 6))
        sns.countplot(y=feature, data=df, order=value_counts.index)
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
        plt.show()
    
    # For numeric features
    if np.issubdtype(series.dtype, np.number):
        print("\n--- Numeric Analysis ---")
        
        # Basic statistics
        stats_df = pd.DataFrame({
            'Statistic': ['count', 'mean', 'std', 'min', '25%', '50% (median)', '75%', 'max', 
                         'skewness', 'kurtosis', 'IQR', 'range'],
            'Value': [
                series.count(),
                series.mean(),
                series.std(),
                series.min(),
                series.quantile(0.25),
                series.median(),
                series.quantile(0.75),
                series.max(),
                series.skew(),
                series.kurtosis(),
                series.quantile(0.75) - series.quantile(0.25),
                series.max() - series.min()
            ]
        })
        print(stats_df)
        
        # Check for outliers using IQR method
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        print(f"\nOutliers (IQR method): {len(outliers)} ({len(outliers)/len(series)*100:.2f}% of non-null values)")
        print(f"Outlier bounds: Lower < {lower_bound:.2f} or Upper > {upper_bound:.2f}")
        
        if len(outliers) > 0 and len(outliers) < 20:
            print("Outlier values:")
            print(outliers.values)
        
        # Normality test (Shapiro-Wilk)
        if series.count() > 3 and series.count() <= 5000:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, p_value = stats.shapiro(series.dropna())
                    print(f"\nShapiro-Wilk Normality Test: statistic={stat:.4f}, p-value={p_value:.4f}")
                    alpha = 0.05
                    if p_value > alpha:
                        print(f"Feature appears normally distributed (p > {alpha})")
                    else:
                        print(f"Feature does not appear normally distributed (p < {alpha})")
            except:
                print("Couldn't perform Shapiro-Wilk normality test")
        elif series.count() > 5000:
            print("\nSkipping Shapiro-Wilk normality test due to large sample size (n > 5000).")
            
        # Visualizations
        plt.figure(figsize=(15, 10))
        
        # Histogram
        plt.subplot(2, 2, 1)
        sns.histplot(series, kde=True)
        plt.title(f'Distribution of {feature}')
        
        # Box plot
        plt.subplot(2, 2, 2)
        sns.boxplot(y=series)
        plt.title(f'Box Plot of {feature}')
        
        # Q-Q plot
        plt.subplot(2, 2, 3)
        stats.probplot(series.dropna(), dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {feature}')
        
        # KDE plot
        plt.subplot(2, 2, 4)
        sns.kdeplot(series)
        plt.title(f'KDE Plot of {feature}')
        
        plt.tight_layout()
        plt.show()
    
    # Time series analysis if timestamp
    if pd.api.types.is_datetime64_any_dtype(series) or feature.lower() in ['date', 'time', 'timestamp', 'datetime']:
        print("\n--- Time Series Analysis ---")
        
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(series):
                series = pd.to_datetime(series, errors='coerce')
                
            print(f"Date range: {series.min()} to {series.max()}")
            print(f"Time span: {series.max() - series.min()}")
            
            # Count by year, month, and day of week
            if (series.max() - series.min()).days > 365:
                yearly_counts = series.dt.year.value_counts().sort_index()
                print("\nCounts by year:")
                print(yearly_counts)
            
            monthly_counts = series.dt.month.value_counts().sort_index()
            print("\nCounts by month:")
            print(monthly_counts)
            
            dow_counts = series.dt.day_name().value_counts()
            print("\nCounts by day of week:")
            print(dow_counts)
            
            # Plot time series
            plt.figure(figsize=(12, 8))
            
            # By month
            plt.subplot(2, 1, 1)
            sns.countplot(x=series.dt.month, order=range(1, 13))
            plt.title(f'Distribution of {feature} by Month')
            plt.xlabel('Month')
            
            # By day of week
            plt.subplot(2, 1, 2)
            sns.countplot(y=series.dt.day_name(), order=['Monday', 'Tuesday', 'Wednesday', 
                                                       'Thursday', 'Friday', 'Saturday', 'Sunday'])
            plt.title(f'Distribution of {feature} by Day of Week')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Couldn't perform time series analysis: {e}")
    
    # Correlation with other numeric features
    if np.issubdtype(series.dtype, np.number) and len(df.select_dtypes(include=[np.number]).columns) > 1:
        print("\n--- Correlation with Other Numeric Features ---")
        
        # Calculate correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()[feature].sort_values(ascending=False)
        
        # Remove self-correlation
        correlations = correlations[correlations.index != feature]
        
        corr_df = pd.DataFrame({
            'Feature': correlations.index,
            'Pearson Correlation': correlations.values
        })
        print(corr_df)
        
        # Plot top and bottom correlations
        if len(correlations) > 0:
            plt.figure(figsize=(10, 6))
            
            # Get top 10 and bottom 10 correlations
            top_corr = correlations.abs().sort_values(ascending=False).head(10).index
            
            # Create correlation plot for selected features
            if len(top_corr) > 0:
                corr_data = df[[feature] + list(top_corr)].corr()
                mask = np.triu(np.ones_like(corr_data, dtype=bool))
                sns.heatmap(corr_data, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
                plt.title(f'Correlation of {feature} with Other Features')
                plt.tight_layout()
                plt.show()
                
                # Scatter plots for the top 3 correlations
                if len(top_corr) >= 3:
                    plt.figure(figsize=(15, 5))
                    for i, col in enumerate(top_corr[:3]):
                        plt.subplot(1, 3, i+1)
                        sns.scatterplot(x=col, y=feature, data=df)
                        plt.title(f'{feature} vs {col}')
                    plt.tight_layout()
                    plt.show()
                
    # Return the analysis as a dictionary
    analysis_summary = {
        'feature': feature,
        'dtype': str(series.dtype),
        'missing_count': null_count,
        'missing_percent': null_percent,
        'unique_values': series.nunique(),
    }
    
    if np.issubdtype(series.dtype, np.number):
        analysis_summary.update({
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'has_outliers': len(outliers) > 0,
            'outliers_count': len(outliers)
        })

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



def prepare_clean_dataset(df, target=None, features=None):
    """
    Create a new DataFrame with only rows that have non-missing values for both specified features.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        feature1 (str): First feature name to check for missing values
        feature2 (str): Second feature name to check for missing values
        
    Returns:
        pandas.DataFrame: Clean dataset with no missing values in both specified features
    """
    if features is not None :
        clean_df = df.dropna(subset=features)
    else :
        clean_df = df.dropna(subset=[target])
  
    
    
    # Report statistics
    original_count = len(df)
    clean_count = len(clean_df)
    
    print(f"Original dataset: {original_count} rows")
    print(f"Clean dataset: {clean_count} rows ({clean_count/original_count:.2%} of original)")
    print(f"\nMissing values summary:")
    print(f"Rows missing either feature: {original_count - clean_count}")
    
    return clean_df


def get_nan_rows(dataframe, features):
    """
    Return all rows where the specified feature(s) has NaN values.
    
    Args:
        dataframe: pandas DataFrame
        features: str or list of str - column name(s) to check for NaN values
    
    Returns:
        pandas DataFrame containing only rows where at least one of the specified features has NaN
    """
    # Convert single feature to list for consistent handling
    if isinstance(features, str):
        features = [features]
    
    # Validate that all features exist in the dataframe
    missing_features = [f for f in features if f not in dataframe.columns]
    if missing_features:
        raise ValueError(f"Features not found in dataframe: {', '.join(missing_features)}")
    
    # Create a mask for rows where any of the specified features is NaN
    nan_mask = dataframe[features].isna().any(axis=1)
    
    # Get the count of NaN rows
    nan_count = nan_mask.sum()
    
    # Filter the dataframe using the mask
    nan_rows = dataframe[nan_mask].copy()
    
    # Print summary
    print(f"Found {nan_count} rows with NaN values in features: {', '.join(features)}")
    print(f"This represents {(nan_count / len(dataframe) * 100):.2f}% of the dataframe ({nan_count} out of {len(dataframe)} rows)")
    
    # Count NaNs per feature
    if len(features) > 1:
        print("\nNaN count by feature:")
        for feature in features:
            feature_nan_count = dataframe[feature].isna().sum()
            print(f"  - {feature}: {feature_nan_count} NaN values ({(feature_nan_count / len(dataframe) * 100):.2f}%)")
    
    return nan_rows


def introduce_missingness(df, feature1, missing_rate, pattern='MCAR', feature2=None, task='regression', seed=42):
    """
    Introduce missing values in a specific feature of a DataFrame with a given pattern and rate,
    while saving the original values paired with appropriate time or ID index.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        feature1 (str): Name of the feature to introduce missing values
        missing_rate (float): Rate of missing values to introduce (between 0 and 1)
        pattern (str): Pattern of missingness ('MCAR', 'MAR', or 'MNAR')
        feature2 (str, optional): Name of the target variable (needed for MAR pattern)
        task (str): Specifies the task type - 'regression' (uses 'charttime' as index) or 
                   'classification' (uses 'hadm_id' as index)
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (df_with_missing, ground_truth) where:
            - df_with_missing: DataFrame with missing values introduced
            - ground_truth: DataFrame with only original values and appropriate index
    """
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Check if feature exists in DataFrame
    if feature1 not in df.columns:
        raise ValueError(f"Feature '{feature1}' not found in DataFrame columns: {df.columns.tolist()}")
    
    # Determine index column based on task type (same for both)
    index_col = 'subject_id' 
    
    # Check if index column exists
    if index_col not in df.columns:
        raise ValueError(f"Column '{index_col}' not found in DataFrame. Required for task='{task}'")
    
    # Check if we need target for MAR pattern
    if pattern == 'MAR' and (feature2 is None or feature2 not in df.columns):
        raise ValueError(f"Target variable is required for MAR pattern. Target '{feature2}' not found.")
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Create a ground truth DataFrame with only original values and the appropriate index
    ground_truth = pd.DataFrame(index=df_copy.index)
    ground_truth['original_values'] = df_copy[feature1].copy()
    ground_truth[index_col] = df_copy[index_col].copy()
    
    # Check for existing NaN values
    original_mask = df_copy[feature1].isna()
    existing_nan_count = original_mask.sum()
    n_samples = len(df_copy)
    existing_nan_rate = existing_nan_count / n_samples
    
    # Calculate how many values to make missing
    if existing_nan_rate > 0:
        print(f"Warning: Feature '{feature1}' already has {existing_nan_rate:.2%} missing values.")
        print("Using only rows with existing values to introduce new missingness.")
    
    # Focus only on non-missing values for introducing missingness
    valid_mask = ~df_copy[feature1].isna()
    valid_indices = df_copy.index[valid_mask].tolist()
    
    if len(valid_indices) == 0:
        raise ValueError(f"Feature '{feature1}' has no valid values. Cannot introduce missingness.")
    
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
            'target_value': df_copy.loc[valid_indices, feature2].values
        })
        
        # Sort by target values and select indices with lowest target values
        target_df = target_df.sort_values('target_value')
        missing_indices = target_df['index'].head(n_to_make_missing).tolist()
    
    elif pattern == 'MNAR':
        # Missing Not At Random - depends on feature value itself
        # Create a DataFrame with valid indices and feature values
        feature_df = pd.DataFrame({
            'index': valid_indices,
            'feature_value': df_copy.loc[valid_indices, feature1].values
        })
        
        # Sort by feature values and select indices with lowest feature values
        feature_df = feature_df.sort_values('feature_value')
        missing_indices = feature_df['index'].head(n_to_make_missing).tolist()
    
    else:
        raise ValueError("Pattern must be one of: 'MCAR', 'MAR', 'MNAR'")
    
    # Introduce missing values in the copy
    df_copy.loc[missing_indices, feature1] = np.nan
    
    # Calculate final missing rate for verification
    final_missing_count = df_copy[feature1].isna().sum()
    final_missing_rate = final_missing_count / n_samples
    
    print(f"Task: {task} (using {index_col} as index)")
    print(f"Missingness pattern: {pattern}")
    print(f"Original missing rate: {existing_nan_rate:.2%}")
    print(f"Added {n_to_make_missing} missing values")
    print(f"Final missing rate: {final_missing_rate:.2%}")
    
    return df_copy, ground_truth


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

class LinearVFLModel(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Create a proper linear layer that transforms inputs to class logits
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim, bias=True)
        )
        # Initialize with small random weights
        torch.nn.init.xavier_uniform_(self.model[0].weight)
    
    def forward(self, x):
        # Apply the linear transformation
        return self.model(x)


class ClientEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=4):
        super(ClientEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)
  

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


def load_vfl_model(model_path):
    """
    Load a VFL model with the specific directory structure:
    - model_path/
      - final_model_state_dict.pth
      - final_model.pth
      - model_info.json
      - encoders/
        - client_encoder_1.pth, etc.
      - nodes_preprocessor/
        - preprocessor_1.pkl, etc.
      - server_preprocessor/
        - label_map.pkl (for classification)
        - target_scaler.pkl (for regression)
        
    Args:
        model_path: Path to the server model directory
        
    Returns:
        server_model: The loaded server model
        client_encoders: Dictionary of client encoders
        model_artifacts: Dictionary containing preprocessors and metadata
    """
    # Check if model path exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Determine if regression or classification
    is_regression = "regression" in model_path
    model_type = None
    
    if "dl_regression" in model_path:
        model_type = "dl_r"
    elif "ml_regression" in model_path:
        model_type = "ml_r"
    elif "dl_classification" in model_path:
        model_type = "dl_c" 
    elif "ml_classification" in model_path:
        model_type = "ml_c"
    
    # Load model info from model_info.json
    model_info_path = os.path.join(model_path, "model_info.json")
    input_dim = 20  # Default
    output_dim = 1  # Default
    
    if os.path.exists(model_info_path):
        try:
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
                input_dim = model_info.get("input_dim", 20)
                output_dim = model_info.get("output_dim", 1)
            print(f"Loaded model info from {model_info_path}")
        except Exception as e:
            print(f"Error loading model info: {e}")
            model_info = {}
    else:
        print(f"No model_info.json found at {model_path}")
        model_info = {}
    
    # Create appropriate model instance
    if is_regression:
        server_model = SimpleRegressor(input_dim=input_dim)
    else:
        server_model = SimpleClassifier(input_dim=input_dim, num_classes=output_dim)
    
    # Load the model - try state_dict first, then full model
    state_dict_path = os.path.join(model_path, "final_model_state_dict.pth")
    if os.path.exists(state_dict_path):
        try:
            server_model.load_state_dict(torch.load(state_dict_path))
            print(f"Loaded model state_dict from {state_dict_path}")
        except Exception as e:
            print(f"Error loading state_dict: {e}")
            
            # Try loading full model with weights_only=False
            full_model_path = os.path.join(model_path, "final_model.pth")
            if os.path.exists(full_model_path):
                try:
                    add_safe_globals([SimpleRegressor, SimpleClassifier])
                    server_model = torch.load(full_model_path, weights_only=False)
                    print(f"Loaded full model from {full_model_path}")
                except Exception as e2:
                    print(f"Error loading full model: {e2}")
    else:
        # No state_dict, try full model
        full_model_path = os.path.join(model_path, "final_model.pth")
        if os.path.exists(full_model_path):
            try:
                from torch.serialization import add_safe_globals
                add_safe_globals([SimpleRegressor, SimpleClassifier])
                server_model = torch.load(full_model_path, weights_only=False)
                print(f"Loaded full model from {full_model_path}")
            except Exception as e:
                print(f"Error loading full model: {e}")
        else:
            print(f"WARNING: No model file found at {model_path}")
    
    # Load client encoders from encoders/ directory
    client_encoders = {}
    encoder_dir = os.path.join(model_path, "encoders")
    
    if os.path.exists(encoder_dir):
        print(f"Loading client encoders from {encoder_dir}")
        for file in os.listdir(encoder_dir):
            if file.endswith(".pth"):
                # Extract node_id from filename
                if file.startswith("client_encoder_"):
                    node_id = file.split("_")[-1].split(".")[0]
                else:
                    # Try to extract a number from the filename
                    import re
                    match = re.search(r'(\d+)', file)
                    if match:
                        node_id = match.group(1)
                    else:
                        # Use the filename without extension as node_id
                        node_id = os.path.splitext(file)[0]
                
                encoder_path = os.path.join(encoder_dir, file)
                try:
                    from torch.serialization import add_safe_globals
                    add_safe_globals([ClientEncoder])
                    client_encoders[node_id] = torch.load(encoder_path, weights_only=False)
                    print(f"Loaded encoder for node {node_id}")
                except Exception as e:
                    print(f"Error loading encoder for node {node_id}: {e}")
    else:
        print(f"No encoders directory found at {encoder_dir}")
    
    # Load node preprocessors from nodes_preprocessor/ directory
    node_preprocessors = {}
    preprocessor_dir = os.path.join(model_path, "nodes_preprocessor")
    
    if os.path.exists(preprocessor_dir):
        print(f"Loading node preprocessors from {preprocessor_dir}")
        for file in os.listdir(preprocessor_dir):
            if file.endswith(".pkl"):
                # Extract node_id from filename
                if file.startswith("preprocessor_"):
                    node_id = file.split("_")[-1].split(".")[0]
                else:
                    # Try to extract a number from the filename
                    import re
                    match = re.search(r'(\d+)', file)
                    if match:
                        node_id = match.group(1)
                    else:
                        # Use the filename without extension as node_id
                        node_id = os.path.splitext(file)[0]
                
                preprocessor_path = os.path.join(preprocessor_dir, file)
                try:
                    with open(preprocessor_path, "rb") as f:
                        node_preprocessors[node_id] = pickle.load(f)
                    print(f"Loaded preprocessor for node {node_id}")
                except Exception as e:
                    print(f"Error loading preprocessor for node {node_id}: {e}")
    else:
        print(f"No nodes_preprocessor directory found at {preprocessor_dir}")
    
    # Load server preprocessor (label_map or target_scaler)
    server_preprocessor_dir = os.path.join(model_path, "server_preprocessor")
    label_map = None
    target_scaler = None
    
    if os.path.exists(server_preprocessor_dir):
        print(f"Loading server preprocessor from {server_preprocessor_dir}")
        
        # For classification, look for label_map.pkl
        if not is_regression:
            label_map_path = os.path.join(server_preprocessor_dir, "label_map.pkl")
            if os.path.exists(label_map_path):
                try:
                    with open(label_map_path, "rb") as f:
                        label_map = pickle.load(f)
                    print(f"Loaded label map with {len(label_map)} classes")
                except Exception as e:
                    print(f"Error loading label map: {e}")
        
        # For regression, look for target_scaler.pkl
        if is_regression:
            scaler_path = os.path.join(server_preprocessor_dir, "target_scaler.pkl")
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, "rb") as f:
                        target_scaler = pickle.load(f)
                    print("Loaded target scaler")
                except Exception as e:
                    print(f"Error loading target scaler: {e}")
    else:
        print(f"No server_preprocessor directory found at {server_preprocessor_dir}")
    
    # Combine all artifacts
    model_artifacts = {
        "model_info": model_info,
        "model_type": model_type,
        "is_regression": is_regression,
        "node_preprocessors": node_preprocessors,
        "label_map": label_map,
        "target_scaler": target_scaler
    }
    
    return server_model, client_encoders, model_artifacts




def load_ml_model(model_path, verbose=True):
    """
    Load an ML VFL model with weights and preprocessors
    
    Args:
        model_path: Path to the model directory (e.g., "../server/results/ml_regression")
        verbose: Whether to print status messages
        
    Returns:
        Dictionary containing all artifacts needed for prediction
    """
  

    # Add classes to safe globals
    add_safe_globals([LinearVFLModel])
    
    # Check if model path exists
    if not os.path.exists(model_path):
        if verbose:
            print(f"Model path does not exist: {model_path}")
        return {"error": "Model path not found"}
    
    # Determine if regression or classification
    is_regression = "regression" in model_path
    model_type = None
    
    if "ml_regression" in model_path:
        model_type = "ml_r"
    elif "ml_classification" in model_path:
        model_type = "ml_c"
    else:
        if verbose:
            print(f"Path {model_path} is not recognized as an ML model path")
        return {"error": "Invalid model path"}
    
    # Initialize result dictionary
    ml_artifacts = {
        "node_weights": {},
        "node_preprocessors": {},
        "node_metadata": {},
        "client_features": {},
        "is_regression": is_regression,
        "model_type": model_type
    }
    
    # Load model info
    model_info_path = os.path.join(model_path, "model_info.json")
    if os.path.exists(model_info_path):
        try:
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
            ml_artifacts["model_info"] = model_info
            if verbose:
                print(f"Loaded model info from {model_info_path}")
        except Exception as e:
            if verbose:
                print(f"Error loading model info: {e}")
            ml_artifacts["model_info"] = {}
    else:
        if verbose:
            print(f"No model_info.json found at {model_path}")
        ml_artifacts["model_info"] = {}
    
    # Create a default model
    server_model = LinearVFLModel(input_dim=2, output_dim=4)
    
    # Try to load saved model
    state_dict_path = os.path.join(model_path, "final_model_state_dict.pth")
    if os.path.exists(state_dict_path):
        try:
            server_model.load_state_dict(torch.load(state_dict_path))
            if verbose:
                print(f"Loaded model state_dict from {state_dict_path}")
        except Exception as e:
            if verbose:
                print(f"Error loading state_dict: {e}")
            
            # Try loading full model
            full_model_path = os.path.join(model_path, "final_model.pth")
            if os.path.exists(full_model_path):
                try:
                    server_model = torch.load(full_model_path, weights_only=False)
                    if verbose:
                        print(f"Loaded full model from {full_model_path}")
                except Exception as e2:
                    if verbose:
                        print(f"Error loading full model: {e2}")
    else:
        # No state_dict, try full model
        full_model_path = os.path.join(model_path, "final_model.pth")
        if os.path.exists(full_model_path):
            try:
                server_model = torch.load(full_model_path, weights_only=False)
                if verbose:
                    print(f"Loaded full model from {full_model_path}")
            except Exception as e:
                if verbose:
                    print(f"Error loading full model: {e}")
        else:
            if verbose:
                print(f"WARNING: No model file found at {model_path}")
    
    ml_artifacts["server_model"] = server_model
    
    # Load node weights from weights/ directory
    weights_dir = os.path.join(model_path, "weights")
    if os.path.exists(weights_dir):
        if verbose:
            print(f"Loading node weights from {weights_dir}")
        for file in os.listdir(weights_dir):
            if file.endswith(".pth") and file.startswith("weights_"):
                # Extract node_id from filename
                node_id = file.split("_")[-1].split(".")[0]
                weight_path = os.path.join(weights_dir, file)
                try:
                    ml_artifacts["node_weights"][node_id] = torch.load(weight_path)
                    if verbose:
                        print(f"Loaded weights for node {node_id}")
                        
                    # Try to load metadata for this node
                    metadata_path = os.path.join(weights_dir, f"weights_metadata_{node_id}.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            node_metadata = json.load(f)
                            ml_artifacts["node_metadata"][node_id] = node_metadata
                            
                            # Extract features for convenience
                            if "features" in node_metadata:
                                ml_artifacts["client_features"][node_id] = node_metadata["features"]
                except Exception as e:
                    if verbose:
                        print(f"Error loading weights for node {node_id}: {e}")
    else:
        if verbose:
            print(f"No weights directory found at {weights_dir}")
    
    # Load node preprocessors from nodes_preprocessor/ directory
    preprocessor_dir = os.path.join(model_path, "nodes_preprocessor")
    if os.path.exists(preprocessor_dir):
        if verbose:
            print(f"Loading node preprocessors from {preprocessor_dir}")
        for file in os.listdir(preprocessor_dir):
            if file.endswith(".pkl") and file.startswith("preprocessor_"):
                node_id = file.split("_")[-1].split(".")[0]
                preprocessor_path = os.path.join(preprocessor_dir, file)
                try:
                    with open(preprocessor_path, "rb") as f:
                        ml_artifacts["node_preprocessors"][node_id] = pickle.load(f)
                    if verbose:
                        print(f"Loaded preprocessor for node {node_id}")
                except Exception as e:
                    if verbose:
                        print(f"Error loading preprocessor for node {node_id}: {e}")
    else:
        if verbose:
            print(f"No nodes_preprocessor directory found at {preprocessor_dir}")
    
    # Load server preprocessor (label_map or target_scaler)
    server_preprocessor_dir = os.path.join(model_path, "server_preprocessor")
    if os.path.exists(server_preprocessor_dir):
        if verbose:
            print(f"Loading server preprocessor from {server_preprocessor_dir}")
        
        if is_regression:
            # For regression, load target_scaler.pkl
            scaler_path = os.path.join(server_preprocessor_dir, "target_scaler.pkl")
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, "rb") as f:
                        ml_artifacts["target_scaler"] = pickle.load(f)
                    if verbose:
                        print("Loaded target scaler")
                        
                    # Check if scaler is fitted
                    scaler = ml_artifacts["target_scaler"]
                    if hasattr(scaler, "transformers_") or hasattr(scaler, "transformers"):
                        attr = "transformers_" if hasattr(scaler, "transformers_") else "transformers"
                        transformers = getattr(scaler, attr)
                        name, transformer, cols = transformers[0]
                        if hasattr(transformer, 'scale_') and hasattr(transformer, 'mean_'):
                            if verbose:
                                print(f"Target scaler transformer '{name}' is properly fitted")
                        else:
                            if verbose:
                                print(f"Warning: Target scaler transformer '{name}' is not fitted")
                    elif hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_'):
                        if verbose:
                            print("Target scaler is properly fitted")
                    else:
                        if verbose:
                            print("Warning: Target scaler does not appear to be fitted")
                except Exception as e:
                    if verbose:
                        print(f"Error loading target scaler: {e}")
        else:
            # For classification, load label_map.pkl
            label_map_path = os.path.join(server_preprocessor_dir, "label_map.pkl")
            if os.path.exists(label_map_path):
                try:
                    with open(label_map_path, "rb") as f:
                        ml_artifacts["label_map"] = pickle.load(f)
                    if verbose:
                        print(f"Loaded label map with {len(ml_artifacts['label_map'])} classes")
                except Exception as e:
                    if verbose:
                        print(f"Error loading label map: {e}")
    else:
        if verbose:
            print(f"No server_preprocessor directory found at {server_preprocessor_dir}")
    
    # Validate loaded artifacts
    if not ml_artifacts["node_weights"]:
        if verbose:
            print("Warning: No node weights were loaded. Predictions may not be possible.")
    
    return ml_artifacts


def predict_dl(df, approche, client_features=None):
    """
    Process a dataframe through a VFL model and add predictions
    
    Args:
        df: Pandas DataFrame to process
        approche: The model approach ('dl_r', 'dl_c', 'ml_r', 'ml_c')
        client_features: Optional dict mapping node_id to feature lists
                        If None, will try to infer from client encoders
    
    Returns:
        Tuple of (df, predictions) where predictions are the raw prediction values
    """
    result_df = df.copy()
    predictions = None
    
    if approche == 'dl_r':
        model_path = "../server/results/dl_regression"
    elif approche == 'dl_c':
        model_path = "../server/results/dl_classification"
    elif approche == 'ml_r':
        model_path = "../server/results/ml_regression"
    elif approche == 'ml_c':
        model_path = "../server/results/ml_classification"
    else:
        raise ValueError(f"Unknown model approach: {approche}")

    # Load model, encoders and artifacts
    server_model, client_encoders, artifacts = load_vfl_model(model_path)
    
    if not client_encoders:
        print("No client encoders found. Cannot make predictions.")
        return result_df, predictions
    
    # Set models to evaluation mode
    server_model.eval()
    for encoder in client_encoders.values():
        encoder.eval()
    
    # Extract key artifacts
    is_regression = artifacts.get("is_regression", True)
    node_preprocessors = artifacts.get("node_preprocessors", {})
    central_preprocessor = artifacts.get("central_preprocessor")
    label_map = artifacts.get("label_map")
    target_scaler = artifacts.get("target_scaler")
    
    # If client_features not provided, try to get from encoders
    if client_features is None:
        client_features = {}
        for node_id, encoder in client_encoders.items():
            if hasattr(encoder, 'features'):
                client_features[node_id] = encoder.features
    
    print(f"--------------- Retrieved features are {client_features}")
    
    # If we still don't have features for all clients, warn
    missing_nodes = [node_id for node_id in client_encoders.keys() 
                    if node_id not in client_features or not client_features[node_id]]
    
    if missing_nodes:
        print(f"Warning: Missing features for nodes: {missing_nodes}")
        print("Will attempt to process other nodes.")
    
    # Process data for each client
    client_data = {}
    for node_id, encoder in client_encoders.items():
        # Skip nodes with no feature information
        if node_id not in client_features or not client_features[node_id]:
            print(f"Skipping node {node_id} - no feature information available")
            continue
        
        features = client_features[node_id]
        
        # Check if all features exist in the dataframe
        missing_features = [f for f in features if f not in result_df.columns]
        if missing_features:
            print(f"Warning: Missing features for node {node_id}: {missing_features}")
            print(f"Available columns: {result_df.columns.tolist()}")
            continue
        
        # Extract features for this node
        node_df = result_df[features].copy()
        
        # Handle missing values
        num_features = [f for f in features if node_df[f].dtype in ['int64', 'float64']]
        for col in num_features:
            if node_df[col].isna().any():
                col_mean = node_df[col].mean()
                node_df[col].fillna(col_mean, inplace=True)
        
        cat_features = [f for f in features if node_df[f].dtype not in ['int64', 'float64']]
        for col in cat_features:
            if node_df[col].isna().any():
                mode_val = node_df[col].mode()[0]
                node_df[col].fillna(mode_val, inplace=True)
        
        # Preprocess using saved preprocessor or create a new one
        try:
            if node_id in node_preprocessors:
                # Use node-specific preprocessor
                X = node_preprocessors[node_id].transform(node_df)
                print(f"Used node-specific preprocessor for node {node_id}")
            elif central_preprocessor:
                # Use central preprocessor
                X = central_preprocessor.transform(node_df)
                print(f"Used central preprocessor for node {node_id}")
            else:
                # Create a new preprocessor
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import StandardScaler, OneHotEncoder
                node_preprocessor = ColumnTransformer([
                    ('num', StandardScaler(), num_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
                ])
                X = node_preprocessor.fit_transform(node_df)
                print(f"Created new preprocessor for node {node_id}")
            
            # Convert to dense if sparse
            if hasattr(X, "toarray"):
                X = X.toarray()
            
            # Convert to tensor
            X_tensor = torch.tensor(X, dtype=torch.float32)
            
            # Replace NaN values if any
            if torch.isnan(X_tensor).any():
                X_tensor = torch.nan_to_num(X_tensor, nan=0.0)
            
            client_data[node_id] = X_tensor
            
        except Exception as e:
            print(f"Error processing data for node {node_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Make predictions if we have data for at least one client
    if not client_data:
        print("No valid client data processed. Cannot make predictions.")
        return result_df, predictions
    
    # Generate embeddings and make predictions
    with torch.no_grad():
        # Generate embeddings for each client
        embeddings = []
        for node_id in sorted(client_data.keys()):
            encoder = client_encoders[node_id]
            embedding = encoder(client_data[node_id])
            embeddings.append(embedding)
        
        # Concatenate embeddings
        combined_embedding = torch.cat(embeddings, dim=1)
        
        # Get predictions from server model
        model_predictions = server_model(combined_embedding)
        
        # Process predictions based on model type
        if is_regression:
            # Regression - get raw predictions
            pred_values = model_predictions.numpy().flatten()
            
            # Apply inverse transform if target scaler available
            if target_scaler:
                try:
                    # If target_scaler is a ColumnTransformer
                    if hasattr(target_scaler, 'transformers_'):
                        # Extract the actual transformer (StandardScaler) from ColumnTransformer
                        name, transformer, columns = target_scaler.transformers_[0]
                        # Use the actual transformer for inverse_transform
                        pred_values = transformer.inverse_transform(
                            pred_values.reshape(-1, 1)).flatten()
                        print(f"Applied inverse scaling using {name} transformer")
                    else:
                        # If it's a direct scaler like StandardScaler
                        pred_values = target_scaler.inverse_transform(
                            pred_values.reshape(-1, 1)).flatten()
                        print("Applied inverse scaling to predictions")
                except Exception as e:
                    print(f"Error applying inverse scaling: {e}")
                    
                    # Fallback: Try manual inverse scaling for anchor_age prediction
                    try:
                        # Extract basic statistics from the dataframe to estimate scaling
                        if 'anchor_age' in result_df.columns:
                            mean_age = result_df['anchor_age'].mean()
                            std_age = result_df['anchor_age'].std()
                            
                            # Apply manual inverse transformation
                            pred_values = pred_values * std_age + mean_age
                            print(f"Applied manual inverse scaling using dataset statistics: mean={mean_age:.2f}, std={std_age:.2f}")
                        else:
                            print("Could not apply manual scaling - target column not in dataframe")
                    except Exception as e2:
                        print(f"Manual scaling also failed: {e2}")
            
            # Store predictions to return, don't modify dataframe
            predictions = pred_values
        else:
            # Classification - handle probabilities and predicted class
            logits = model_predictions.numpy()
            probs = torch.softmax(model_predictions, dim=1).numpy()
            pred_classes = np.argmax(probs, axis=1)
            
            # Store predicted class
            result_df['predicted_class'] = pred_classes
            
            # Add probabilities for each class
            for i in range(probs.shape[1]):
                class_name = f"class_{i}"
                if label_map and i < len(label_map):
                    class_name = label_map[i]
                result_df[f'prob_{class_name}'] = probs[:, i]
            
            # Map to labels if available
            if label_map:
                predictions = [label_map[idx] if idx < len(label_map) else f"Unknown-{idx}" 
                              for idx in pred_classes]
            else:
                predictions = [f"Class_{idx}" for idx in pred_classes]
            
            # Store the predicted labels
            result_df['predicted_label'] = predictions
    
    return result_df, predictions

def predict_with_ml_model(df, approche, client_features=None, verbose=True):
    """
    Unified prediction function that handles both ML regression and classification models
    
    Args:
        df: Pandas DataFrame with data to predict
        approche: 'ml_r' or 'ml_c' for ML regression/classification
        client_features: Optional dict mapping node_id to feature lists
        verbose: Whether to print status messages
        
    Returns:
        Tuple of (df, predictions) where predictions are the raw values
    """
    result_df = df.copy()
    predictions = None
    
    # Map approach to model path
    model_type_mapping = {
        "ml_r": "ml_regression",
        "ml_c": "ml_classification"
    }
    
    if approche not in model_type_mapping:
        raise ValueError(f"Approach '{approche}' not recognized. Use 'ml_r' or 'ml_c'")
    
    model_path = f"../server/results/{model_type_mapping[approche]}"
    
    # Load the ML model artifacts
    ml_artifacts = load_ml_model(model_path, verbose=verbose)
    
    # Check if loading was successful
    if "error" in ml_artifacts:
        if verbose:
            print(f"Error loading model: {ml_artifacts['error']}")
        return result_df, predictions
    
    # Use provided client features or ones loaded from metadata
    if client_features is not None:
        ml_artifacts["client_features"].update(client_features)
    
    # Check if we have enough information to make predictions
    if not ml_artifacts["node_weights"]:
        if verbose:
            print("No valid weights found. Cannot make predictions.")
        return result_df, predictions
    
    # Set server model to evaluation mode
    server_model = ml_artifacts["server_model"]
    server_model.eval()
    
    # Process data for each node
    node_outputs = []
    
    for node_id, weights in sorted(ml_artifacts["node_weights"].items()):
        # Skip nodes with no feature information
        if node_id not in ml_artifacts["client_features"] or not ml_artifacts["client_features"][node_id]:
            if verbose:
                print(f"Skipping node {node_id} - no feature information available")
            continue
        
        features = ml_artifacts["client_features"][node_id]
        
        # Check if all features exist in the dataframe
        missing_features = [f for f in features if f not in result_df.columns]
        if missing_features:
            if verbose:
                print(f"Warning: Missing features for node {node_id}: {missing_features}")
                print(f"Available columns: {result_df.columns.tolist()}")
            continue
        
        # Extract features for this node
        node_df = result_df[features].copy()
        
        # Handle missing values
        num_features = [f for f in features if node_df[f].dtype in ['int64', 'float64']]
        for col in num_features:
            if node_df[col].isna().any():
                col_mean = node_df[col].mean()
                node_df[col].fillna(col_mean, inplace=True)
                if verbose:
                    print(f"Filled NaN values in {col} with mean: {col_mean:.3f}")
        
        cat_features = [f for f in features if node_df[f].dtype not in ['int64', 'float64']]
        for col in cat_features:
            if node_df[col].isna().any():
                mode_val = node_df[col].mode()[0]
                node_df[col].fillna(mode_val, inplace=True)
                if verbose:
                    print(f"Filled NaN values in {col} with mode: {mode_val}")
        
        # Preprocess using saved preprocessor or fallback
        try:
            if node_id in ml_artifacts["node_preprocessors"]:
                # Use node-specific preprocessor
                X = ml_artifacts["node_preprocessors"][node_id].transform(node_df)
                if verbose:
                    print(f"Used node-specific preprocessor for node {node_id}")
            else:
                # Create a new preprocessor if needed
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import StandardScaler, OneHotEncoder
                
                node_preprocessor = ColumnTransformer([
                    ('num', StandardScaler(), num_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
                ])
                X = node_preprocessor.fit_transform(node_df)
                if verbose:
                    print(f"Created new preprocessor for node {node_id}")
            
            # Convert to dense if sparse
            if hasattr(X, "toarray"):
                X = X.toarray()
            
            # Convert to tensor
            X_tensor = torch.tensor(X, dtype=torch.float32)
            
            # Replace NaN values if any
            if torch.isnan(X_tensor).any():
                X_tensor = torch.nan_to_num(X_tensor, nan=0.0)
                if verbose:
                    print(f"Replaced NaN values in node {node_id} tensor with zeros")
            
            # Apply node weights
            z = X_tensor @ weights
            node_outputs.append(z)
            
        except Exception as e:
            if verbose:
                print(f"Error processing data for node {node_id}: {e}")
                import traceback
                traceback.print_exc()
    
    # Make predictions if we have data from at least one node
    if not node_outputs:
        if verbose:
            print("No valid node outputs processed. Cannot make predictions.")
        return result_df, predictions
    
    # Combine node outputs
    with torch.no_grad():
        # Concatenate the outputs instead of summing them
        combined_output = torch.cat(node_outputs, dim=1)
        
        # Apply server model
        model_predictions = server_model(combined_output)
        
        # Process predictions based on model type
        if ml_artifacts["is_regression"]:
            # Regression - get raw predictions
            pred_values = model_predictions.numpy().flatten() if hasattr(model_predictions, 'numpy') else model_predictions.detach().numpy().flatten()
            
            # Apply inverse transform if target scaler is available
            if "target_scaler" in ml_artifacts:
                target_scaler = ml_artifacts["target_scaler"]
                try:
                    # Different ways to access the actual transformer
                    if hasattr(target_scaler, 'transformers_') or hasattr(target_scaler, 'transformers'):
                        # It's a ColumnTransformer
                        attr_name = 'transformers_' if hasattr(target_scaler, 'transformers_') else 'transformers'
                        transformers = getattr(target_scaler, attr_name)
                        name, transformer, columns = transformers[0]
                        
                        if hasattr(transformer, 'scale_') and hasattr(transformer, 'mean_'):
                            # Transformer is fitted, use it
                            pred_values = transformer.inverse_transform(
                                pred_values.reshape(-1, 1)).flatten()
                            if verbose:
                                print(f"Applied inverse scaling using {name} transformer")
                        else:
                            if verbose:
                                print("Transformer not fitted. Using raw predictions.")
                    elif hasattr(target_scaler, 'scale_') and hasattr(target_scaler, 'mean_'):
                        # Direct scaler like StandardScaler
                        pred_values = target_scaler.inverse_transform(
                            pred_values.reshape(-1, 1)).flatten()
                        if verbose:
                            print("Applied inverse scaling to predictions")
                    else:
                        # Try inverse_transform anyway
                        try:
                            pred_values = target_scaler.inverse_transform(
                                pred_values.reshape(-1, 1)).flatten()
                            if verbose:
                                print("Applied inverse transform to predictions")
                        except Exception as e:
                            if verbose:
                                print(f"Could not apply inverse transform: {e}")
                except Exception as e:
                    if verbose:
                        print(f"Error applying inverse scaling: {e}")
            
            # Store predictions to return
            predictions = pred_values
        else:
            # Classification - get class probabilities and predicted class
            # For ML classification, apply softmax to the raw outputs
            probs = torch.nn.functional.softmax(model_predictions, dim=1).detach().numpy()
            pred_classes = np.argmax(probs, axis=1)
            
            # Map to labels if available
            if "label_map" in ml_artifacts and ml_artifacts["label_map"]:
                label_map = ml_artifacts["label_map"]
                predictions = [label_map[idx] if idx < len(label_map) else f"Unknown-{idx}" 
                              for idx in pred_classes]
            else:
                predictions = [f"Class_{idx}" for idx in pred_classes]
    
    return result_df, predictions

def return_regression_results(dataframe, target=None, client_features=None, verbose=False):
    """
    Run regression prediction pipeline and add results from ML/DL VFL models.
    
    Args:
        dataframe: Input DataFrame
        target: Target column name (optional)
        client_features: Dictionary mapping node_id to feature list (optional)
        verbose: Whether to print status messages
    
    Returns:
        DataFrame with predictions from ML/DL models
    """
    result_df = dataframe.copy()
    
    # Run prediction for ML and DL models
    for model_type in ['ml_r', 'dl_r']:
        try:
            if model_type == 'ml_r':
                # Use the ML approach
                _, ml_predictions = predict_with_ml_model(
                    result_df, 
                    model_type, 
                    client_features=client_features, 
                    verbose=verbose
                )
                if ml_predictions is not None:
                    result_df[f"{target if target else 'prediction'}_ml"] = ml_predictions
                    if verbose:
                        print(f"Added ML regression predictions as '{target if target else 'prediction'}_ml'")
            else:
                # Use the DL approach
                _, dl_predictions = predict_dl(
                    result_df,
                    model_type,
                    client_features=client_features
                )
                if dl_predictions is not None:
                    result_df[f"{target if target else 'prediction'}_dl"] = dl_predictions
                    if verbose:
                        print(f"Added DL regression predictions as '{target if target else 'prediction'}_dl'")
        except Exception as e:
            if verbose:
                print(f"Error during {model_type} prediction: {e}")
                print("Creating empty prediction column")
            result_df[f"{target if target else 'prediction'}_{model_type[:2]}"] = np.nan
    
    # Make sure there's no generic 'prediction' column (which would be a leftover from older versions)
    if 'prediction' in result_df.columns:
        result_df = result_df.drop('prediction', axis=1)
    
    return result_df

def return_classification_results(dataframe, target=None, client_features=None, verbose=False):
    """
    Run classification prediction pipeline and add results from ML/DL models.
    
    Args:
        dataframe: Input DataFrame
        target: Target column name (optional)
        client_features: Dictionary mapping node_id to feature list (optional)
        verbose: Whether to print status messages
    
    Returns:
        DataFrame with predictions from ML/DL models
    """
    result_df = dataframe.copy()
    
    # Run prediction for ML and DL models
    for model_type in ['ml_c', 'dl_c']:
        try:
            if model_type == 'ml_c':
                # Use the ML approach
                temp_df, ml_predictions = predict_with_ml_model(
                    result_df, 
                    model_type, 
                    client_features=client_features, 
                    verbose=verbose
                )
                if ml_predictions is not None:
                    result_df[f"{target if target else 'predicted'}_ml"] = ml_predictions
                    if verbose:
                        print(f"Added ML classification predictions as '{target if target else 'predicted'}_ml'")
            else:
                # Use the DL approach
                temp_df, dl_predictions = predict_dl(
                    result_df,
                    model_type,
                    client_features=client_features
                )
                if dl_predictions is not None:
                    result_df[f"{target if target else 'predicted'}_dl"] = dl_predictions
                    if verbose:
                        print(f"Added DL classification predictions as '{target if target else 'predicted'}_dl'")
        except Exception as e:
            if verbose:
                print(f"Error during {model_type} prediction: {e}")
                print("Creating empty prediction column")
            result_df[f"{target if target else 'predicted'}_{model_type[:2]}"] = None
    
    # Clean up any temporary prediction columns
    for col in ['prediction', 'predicted_class', 'predicted_label']:
        if col in result_df.columns:
            result_df = result_df.drop(col, axis=1)
            
    # Remove probability columns if present (from DL predictions)
    prob_cols = [col for col in result_df.columns if col.startswith('prob_')]
    if prob_cols:
        result_df = result_df.drop(prob_cols, axis=1)
    
    return result_df






